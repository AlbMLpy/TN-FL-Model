from typing import Optional, Callable

import numpy as np

from .features import FeatureMap
from .matrix_operations import khatri_rao_row

def init_weights(
    m_order: int, 
    rank: int, 
    d_dim: int, 
    q_base: Optional[int] = None, 
    init_type: str = 'kj_vec', 
    seed: Optional[int] = None,
    dtype: np.dtype = np.float64,
) -> tuple[np.ndarray, int]:
    """
    Initialize weights in CPD format using Normal Distribution 
    and optional normalization strategies. Use q_base parameter to generate 
    quantized representation of weights.

    Parameters
    ----------
    m_order : int
        The number of new generated features per 1 data feature. 
        In case of quantized version m_order must be a power of 2.

    rank : int
        The rank of the CP Decomposition based weights tensor.

    d_dim : int
        The number of features in the data.

    q_base : int, optional, default=None
        To use a quantized model set q_base=2. 
        To use non-quantized model set q_base=None.

    init_type : str, optional, default='kj_vec'
        The normalization strategy for the weights:
            'k_mtx' - Normalize each matrix in the weights tensor.
            'kj_vec' - Normalize each vector in the matrices of the weights tensor.

    seed : int, optional, default=None
        A seed for the random number generator to ensure reproducibility.

    dtype : np.dtype, default=np.float64
        The data type of the array elements.

    Returns
    -------
    weights : np.ndarray
        An array containing the initialized weights. Shape of array depends on q_base.

    k_d : int
        Degree in the equation: m_order = q_base^(k_d).

    Raises
    ------
    ValueError
        If the `init_type` provided is not recognized.
    """

    if (m_order & (m_order - 1)) and q_base:
        raise ValueError(f"m_order should be a power of 2, but it is {m_order}. ")
    random_state = np.random if seed is None else np.random.RandomState(seed)
    if q_base:
        k_d = int(np.emath.logn(q_base, m_order)) # m_order = q_base^(k_d) 
        weights = random_state.randn(d_dim*k_d, q_base, rank)
    else:
        k_d = 1
        weights = random_state.randn(d_dim, m_order, rank)
    naxis = weights.ndim - 2
    if init_type == 'k_mtx': # Matrix weights[k][:, :] is normalized
        weights /= np.linalg.norm(weights, ord=2, axis=(naxis, naxis + 1), keepdims=True)
    elif init_type == 'kj_vec': # Vector weights[k][:][j] is normalized
        weights /= np.linalg.norm(weights, ord=2, axis=naxis, keepdims=True)
    else:
        raise ValueError(f'Bad init_type = {init_type}. See docs.')
    return weights.astype(dtype), k_d

def get_fw_hadamard_mtx(
    x: np.ndarray, 
    k_d: int,
    weights: np.ndarray,
    feature_map: FeatureMap,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """ 
    Calculate the Hadamard product of matrix multiplication between features and CPD cores.

    Parameters
    ----------
    x : np.ndarray
        Input training data: (n_samples, d_dim).
    
    k_d : int
        Degree in the equation: m_order = q_base^(k_d).

    weights : np.ndarray
        An array containing the initialized weights in CPD format.

    feature_map: FeatureMap
        Mapping from a data feature x_k to new features: f(x_k).

    dtype : np.dtype, default=np.float64
        The data type of the array elements.

    Returns
    -------
    result : np.ndarray
        Array of shape: (n_samples, rank)
    """

    fw_hadamard = 1.0
    for ind, wk in enumerate(weights):
        k, q = divmod(ind, k_d) # q starts from zero -> for feature_map
        fw_hadamard *= feature_map(x[:, k], q).dot(wk)
    return fw_hadamard.astype(dtype)

def get_ww_hadamard_mtx(
    weights: np.ndarray, 
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """ 
    Calculate the Hadamard product of matrix multiplication between corresponding CPD cores.

    Parameters
    ----------
    weights : np.ndarray
        An array containing the initialized weights in CPD format.

    dtype : np.dtype, default=np.float64
        The data type of the array elements.

    Returns
    -------
    result : np.ndarray
        Array of shape: (rank, rank)
    """

    ww_hadamard = 1.0
    for wk in weights:
        ww_hadamard *= wk.T.conj().dot(wk)
    return ww_hadamard.astype(dtype)

def _prepare_system(
    fk_mtx: np.ndarray, 
    fw_hadamard: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    Fk = khatri_rao_row(fw_hadamard, fk_mtx) # Fortran Ordering
    return Fk.T.conj().dot(Fk), Fk.T.conj().dot(y)

def get_updated_als_factor(
    fk_mtx: np.ndarray, 
    fw_hadamard: np.ndarray,
    ww_hadamard: np.ndarray,
    y: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """ 
    Solve custom linear system of equations.
    
    Parameters
    ----------
    fk_mtx : np.ndarray
        Feature matrix: (n_samples, mapping_dim).

    fw_hadamard : int
        Helping Hadamard product matrix of Feature matrix times W_k  CPD core: (n_samples, rank).

    ww_hadamard : int
        Helping Hadamard product matrix of W_k^T@W_k: (rank, rank).

    y : np.ndarray
        Target values array.

    alpha : float
        L2 regularization hyper-parameter.
        
    Returns
    -------
    result : np.ndarray
        Solution of LLS problem for 1 CPD core.
    """

    (_, f_dim), (rank, _) = fk_mtx.shape, ww_hadamard.shape
    A, b = _prepare_system(fk_mtx, fw_hadamard, y)
    if alpha:
        A += alpha * np.kron(ww_hadamard, np.eye(f_dim)) # Fortran Ordering
    return np.linalg.solve(A, b).reshape(f_dim, rank, order='F') # Fortran Ordering

def update_weights(
    x: np.ndarray, 
    y: np.ndarray,
    alpha: float,
    k_d: int,
    weights: np.ndarray,
    feature_map: FeatureMap,
    fw_hadamard: np.ndarray,
    ww_hadamard: np.ndarray,
) -> np.ndarray:
    """ 
    Full update of model weights in CPD format.
    """
    for ind in range(weights.shape[0]):
        # Preprocess:
        k, q = divmod(ind, k_d) # q starts from zero -> for feature_map
        wk, fk_mtx = weights[ind], feature_map(x[:, k], q)
        fw_hadamard /= fk_mtx.dot(wk) 
        ww_hadamard /= wk.T.conj().dot(wk) 
        # Solve linear system:
        weights[ind] = wk = get_updated_als_factor(fk_mtx, fw_hadamard, ww_hadamard, y, alpha)
        # Postprocess:
        fw_hadamard *= fk_mtx.dot(wk)
        ww_hadamard *= wk.T.conj().dot(wk)
    return weights

def predict_score(
    x: np.ndarray, 
    k_d: int, 
    weights: np.ndarray, 
    feature_map: FeatureMap
) -> np.ndarray:
    """ 
    Generate prediction scores for CPD based model. 
    """
    n_samples, rank = x.shape[0], weights.shape[-1]
    score = np.ones((n_samples, rank), dtype=weights.dtype)
    for ind, wk in enumerate(weights):
        k, q = divmod(ind, k_d) # q starts from zero -> for feature_map
        score *= feature_map(x[:, k], q).dot(wk)
    return np.real(np.sum(score, 1))

def run_callback(
    x: np.ndarray, 
    y: np.ndarray, 
    alpha: float, 
    k_d: int, 
    weights: np.ndarray,  
    feature_map: FeatureMap, 
    xy_test: Optional[tuple] = None,
    callback: Optional[Callable] = None,
) -> None:
    """
    Calculates user-defined callback function.
    """
    if callback:
        y_yp = None
        if xy_test:
            x_test, y_test = xy_test
            y_pred_test = predict_score(x_test, k_d, weights, feature_map)
            y_yp = y_test, y_pred_test
        y_pred = predict_score(x, k_d, weights, feature_map)
        callback(dict(y=y, y_pred=y_pred, weights=weights, alpha=alpha, y_yp=y_yp))

def weights3d_to_quantized4d(weights: np.ndarray, k_d: int) -> np.ndarray:
    if k_d <= 1:
        raise ValueError(f"k_d parameter should be > 1. It is: {k_d}")
    d_dim, matrix_shape = weights.shape[0] // k_d, weights.shape[-2:]
    return weights.reshape((d_dim, k_d, *matrix_shape))

def weights4d_quantized_to_3d(weights: np.ndarray) -> tuple[np.ndarray, int]:
    _, k_d, q_base, rank = weights.shape
    return weights.reshape((-1, q_base, rank)), k_d
