from functools import partial
from typing import Optional, Callable

import numpy as np

from .features import FeatureMap
from .matrix_operations import khatri_rao_row
from .model_functionality import (
    init_weights,
    get_fw_hadamard_mtx,
    get_ww_hadamard_mtx,
    predict_score as predict_score_fm,
)
from .cpr import Q_BASE
from .optimization import fista, ls_solution, lsc_solution

def init_feature_weights( 
    n_values: int,
    seed: Optional[int] = None,
    init_equal: bool = False,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    random_state = np.random if seed is None else np.random.RandomState(seed) 
    weights = np.ones(n_values) if init_equal else random_state.rand(n_values) 
    return weights.astype(dtype)

def precalculate_step(
    x: np.ndarray,
    k_d: int,
    weights: np.ndarray,
    fmaps_list: list[FeatureMap],
    dtype: np.dtype = np.float64
) -> tuple[np.ndarray, np.ndarray]:
    shape = (len(fmaps_list), x.shape[0], weights.shape[-1])
    fwh_matrices = np.empty(shape, dtype=weights.dtype)
    for p, feature_map in enumerate(fmaps_list):
        fwh_matrices[p] = get_fw_hadamard_mtx(x, k_d, weights, feature_map, dtype)
    return fwh_matrices, get_ww_hadamard_mtx(weights, dtype)

def prepare_modeling(
    x: np.ndarray,
    m_order: int,
    fmaps_list: list[FeatureMap],
    rank: int,
    init_type: str,
    seed: Optional[int] = None,
    init_equal_lambda: bool = False,
    dtype: np.dtype = np.float64,
) -> tuple:
    n_fmaps, (_, d_dim) = len(fmaps_list), x.shape
    weights, k_d = init_weights(m_order, rank, d_dim, Q_BASE, init_type, seed, dtype)
    lambdas = init_feature_weights(n_fmaps, seed, init_equal_lambda, np.float64)
    fwh_matrices, ww_hadamard = precalculate_step(x, k_d, weights, fmaps_list, dtype)
    return weights, lambdas, k_d, fwh_matrices, ww_hadamard

def update_quantized_weights(
    x: np.ndarray, 
    y: np.ndarray,
    alpha: float,
    k_d: int,
    weights: np.ndarray,
    lambdas: np.ndarray,
    fmaps_list: list[FeatureMap],
    fwh_matrices: np.ndarray,
    ww_hadamard: np.ndarray,
) -> np.ndarray:
    (n, _), (q_base, rank) = x.shape, weights.shape[-2:]
    for ind in range(weights.shape[0]): 
        k, q = divmod(ind, k_d) # q starts from zero -> for feature_map
        wk, Fk = weights[ind], np.zeros((n, q_base*rank), dtype=weights.dtype)
        # Preprocess:
        ww_hadamard /= wk.T.conj().dot(wk)
        for p, feature_map in enumerate(fmaps_list):
            fk_mtx = feature_map(x[:, k], q)    ### Save for better computations? Memory?
            fwh_matrices[p] /= fk_mtx.dot(wk) # remove k-th factor
            if lambdas[p]: # can be zero 
                Fk += lambdas[p] * khatri_rao_row(fwh_matrices[p], fk_mtx) # Fortran Ordering
        # Calculate A, b and solve linear system:
        A, b = Fk.T.conj().dot(Fk), Fk.T.conj().dot(y)
        if alpha:
            A += alpha * np.kron(ww_hadamard, np.eye(q_base)) # Fortran Ordering
        wk = np.linalg.solve(A, b).reshape(q_base, rank, order='F') # Fortran Ordering
        # Postprocess:
        ww_hadamard *= wk.T.conj().dot(wk)
        for p, feature_map in enumerate(fmaps_list):
            fk_mtx = feature_map(x[:, k], q)
            fwh_matrices[p] *= fk_mtx.dot(wk)
        weights[ind] = wk
    return weights, lambdas

def update_feature_weights( 
    x: np.ndarray, 
    y: np.ndarray,
    beta: float,
    k_d: int,
    weights: np.ndarray, 
    lambdas: np.ndarray,
    fmaps_list: list[FeatureMap],
    l_reg: str = 'l2',
    ns_l1: int = 1000,
    l_pos: bool = False,
) -> np.ndarray:
    f_mtx = np.empty((x.shape[0], len(lambdas)))
    for p, feature_map in enumerate(fmaps_list):
        f_mtx[:, p] = predict_score_fm(x, k_d, weights, feature_map)
    if l_reg == 'l2':
        lambdas = ls_solution(f_mtx, y, beta) # fixed l2 norm = 1
        lambdas = np.maximum(lambdas, 0.0) if l_pos else lambdas
    elif l_reg == 'l1':
        if len(np.nonzero(lambdas)[0]) > 1: # do not update if only 1 left
            lambdas = fista(f_mtx, y, lambdas, beta, n_steps=ns_l1, pos=l_pos)
    elif l_reg == 'fixed_norm':
        lambdas = lsc_solution(f_mtx, y, alp=1) # fixed l2 norm = 1
        lambdas = np.maximum(lambdas, 0.0) if l_pos else lambdas
    else:
        raise NotImplementedError("Choose 'l1' or 'l2' or 'fixed_norm'")
    return weights, lambdas

def q_cpr_f(
    x: np.ndarray, 
    y: np.ndarray,
    m_order: int,
    fmaps_list: list[FeatureMap],
    rank: int,
    init_type: str,
    n_epoch: int,
    alpha: float,
    beta: float,
    l_reg: str = 'l2',
    ns_l1: int = 100,
    seed: Optional[int] = None,
    l_init_eq: bool = False,
    l_pos: bool = False,
    dtype: np.dtype = np.float64,
    xy_test: Optional[tuple] = None,
    callback: Optional[Callable] = None,
    upd_t: str = 'wl',
) -> tuple[np.ndarray, np.ndarray, int]:
    weights, lambdas, k_d, fwh_matrices, ww_hadamard = prepare_modeling(
        x, m_order, fmaps_list, rank, init_type, seed, l_init_eq, dtype
    )
    _shared = dict(x=x, y=y, k_d=k_d, fmaps_list=fmaps_list)
    rc = partial(run_callback, alpha=alpha, beta=beta, 
        xy_test=xy_test, callback=callback, **_shared
    )
    rc(weights=weights, lambdas=lambdas)
    for ep in range(n_epoch):
        weights, lambdas = update_feature_weights(weights=weights, lambdas=lambdas, 
            beta=beta, l_reg=l_reg, ns_l1=ns_l1, l_pos=l_pos, **_shared)
        if l_pos and (ep == 0): lambdas /= np.sum(lambdas)
        rc(weights=weights, lambdas=lambdas)
        weights, lambdas = update_quantized_weights(weights=weights, lambdas=lambdas,
            alpha=alpha, fwh_matrices=fwh_matrices, ww_hadamard=ww_hadamard, **_shared)
        rc(weights=weights, lambdas=lambdas)
    weights, lambdas = update_feature_weights(weights=weights, lambdas=lambdas, 
        beta=beta, l_reg=l_reg, ns_l1=ns_l1, l_pos=l_pos, **_shared)
    rc(weights=weights, lambdas=lambdas)
    return weights, lambdas, k_d

def predict_score(
    x: np.ndarray, 
    k_d: int,
    weights: np.ndarray, 
    lambdas: np.ndarray,
    fmaps_list: list[FeatureMap],
) -> np.ndarray:
    score = np.zeros(x.shape[0])
    for p, feature_map in enumerate(fmaps_list):
        if lambdas[p]:
            score += lambdas[p] * predict_score_fm(x, k_d, weights, feature_map)
    return score

def run_callback(
    x, y, k_d: int, weights, lambdas, fmaps_list, alpha, beta, 
    xy_test: Optional[tuple] = None, callback: Optional[Callable] = None
):
    if callback:
        y_yp = None
        if xy_test:
            y_yp = xy_test[1], predict_score(xy_test[0], k_d, weights, lambdas, fmaps_list)
        y_pred = predict_score(x, k_d, weights, lambdas, fmaps_list)
        callback(dict(y=y, y_pred=y_pred, weights=weights, lambdas=lambdas, alpha=alpha, beta=beta, y_yp=y_yp))
