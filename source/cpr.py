from typing import Optional, Callable

import numpy as np

from .features import FeatureMap
from .model_functionality import (
    init_weights,
    get_fw_hadamard_mtx,
    get_ww_hadamard_mtx,
    update_weights,
    run_callback,
)

Q_BASE = 2

def cpr(
    x: np.ndarray, 
    y: np.ndarray,
    quantized: bool, 
    m_order: int,
    feature_map: FeatureMap,
    rank: int,
    init_type: str,
    n_epoch: int,
    alpha: float,
    seed: Optional[int] = None,
    dtype: np.dtype = np.float64,
    xy_test: Optional[tuple] = None,
    callback: Optional[Callable] = None,
) -> tuple[np.ndarray, int]:
    """ 
    Train CPD Regression model using quantized or general format.

    References: 
        [1] "Large-Scale Learning with Fourier Features and Tensor Decompositions", Wesel, Batselier.
        [2] "Quantized Fourier and Polynomial Features for more Expressive Tensor Network Models", Wesel, Batselier.
    
    Parameters
    ----------
    x : numpy.ndarray
        Input data x: (n_samples, n_input_features)

    y : numpy.ndarray
        Target values y: n_samples

    quantized : bool
        Use quantized=True to get quantized representation of weight tensor based on [2]. 
        Use quantized=False to get more general representation of weight tensor based on [1]
    
    m_order : int
        The number of new generated features per 1 data feature. 
        In case of quantized version m_order must be a power of 2.
    
    feature_map: FeatureMap
        Mapping from a data feature x_k to new features: f(x_k).

    rank : int
        The rank of the CP Decomposition based weights tensor.

    init_type : str, optional, default='kj_vec'
        The normalization strategy for the weights:
            'k_mtx' - Normalize each matrix in the weights tensor.
            'kj_vec' - Normalize each vector in the matrices of the weights tensor.

    n_epoch : int
        The number of global parameters updates.

    alpha : float
        L2 regularization hyper-parameter.
        
    seed : int, optional, default=None
        A seed for the random number generator to ensure reproducibility.

    dtype : np.dtype, default=np.float64
        The data type of the array elements.
    
    callback : Optional[Callable] = None
        Function is called before training and after every epoch of training. 
        callback should have the following layout: callback(y, y_pred, weights, **kwargs).
    
    Returns
    -------
    weights : numpy.ndarray
        3D array containing the trained CPD weights.

    k_d : int
        Degree in the equation: m_order = q_base^(k_d).
        
    """
    q_base = Q_BASE if quantized else None
    weights, k_d = init_weights(m_order, rank, x.shape[-1], q_base, init_type, seed, dtype)
    fw_hadamard = get_fw_hadamard_mtx(x, k_d, weights, feature_map, dtype)
    ww_hadamard = get_ww_hadamard_mtx(weights, dtype)
    run_callback(x, y, alpha, k_d, weights, feature_map, xy_test, callback)
    for _ in range(n_epoch):
        weights = update_weights(
            x, y, alpha, k_d, weights, feature_map, fw_hadamard, ww_hadamard)
        run_callback(x, y, alpha, k_d, weights, feature_map, xy_test, callback)
    return weights, k_d
