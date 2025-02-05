import numpy as np

from .model_functionality import get_ww_hadamard_mtx

def mse_metric(y: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y - y_pred)**2)

def rmse_metric(y: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(mse_metric(y, y_pred))

def norm2(x: np.ndarray) -> float:
    return np.linalg.norm(x.flatten(), ord=2)

def normF_cpd(x: np.ndarray) -> float:
    ww_hadamard = get_ww_hadamard_mtx(x, x.dtype)
    return np.sqrt(np.sum(np.real(ww_hadamard)))

def norm1(x: np.ndarray) -> float:
    return np.linalg.norm(x.flatten(), ord=1)

def ls_loss(y: np.ndarray, y_pred: np.ndarray) -> float:
    return 0.5 * norm2(y - y_pred)**2

def l2_reg_cp(weights: np.ndarray) -> float:
    return 0.5 * normF_cpd(weights)**2

def l2_reg(x: np.ndarray) -> float:
    return 0.5 * norm2(x)**2

def ls_l2_loss(
    y: np.ndarray, 
    y_pred: np.ndarray, 
    weights: np.ndarray, 
    alpha: float,
) -> float:
    return ls_loss(y, y_pred) + alpha * l2_reg_cp(weights)

def ls_l1(
    y: np.ndarray, 
    y_pred: np.ndarray, 
    weights: np.ndarray, 
    alpha: float,
) -> float:
    return ls_loss(y, y_pred) + alpha * norm1(weights)

def ls_l2w_l2l_loss(
    y: np.ndarray, 
    y_pred: np.ndarray, 
    weights: np.ndarray, 
    lambdas: np.ndarray,
    alpha: float,
    beta: float,
) -> float:
    return ls_loss(y, y_pred) + alpha * l2_reg_cp(weights) + 0.5 * beta * norm2(lambdas)**2

def ls_l2w_l1l_loss(
    y: np.ndarray, 
    y_pred: np.ndarray, 
    weights: np.ndarray, 
    lambdas: np.ndarray,
    alpha: float,
    beta: float,
) -> float:
    return ls_loss(y, y_pred) + alpha * l2_reg_cp(weights) + beta * norm1(lambdas)
