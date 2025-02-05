import numpy as np
from scipy.linalg import svd
from scipy.optimize import newton

### L1 Regularization optimization ###
def shrinkage_operator(x: np.ndarray, alpha: float) -> np.ndarray:
    return np.sign(x) * np.maximum((np.abs(x) - alpha), 0)

def shrinkage_operator_p(x: np.ndarray, alpha: float) -> np.ndarray:
    return np.maximum((np.abs(x) - alpha), 0)

def fista( 
    x: np.ndarray, 
    y: np.ndarray,
    w: np.ndarray,
    beta: float,
    n_steps: int,
    atol: float = 1e-4,
    pos: bool = False,
) -> np.ndarray:
    a_mtx, b = x.T.conj().dot(x), x.T.conj().dot(y)
    yk, tk_prev = 1*w, 1
    ss = 1 / np.linalg.norm(a_mtx, ord=2)
    n1 = np.linalg.norm(w, ord=2)
    sh = shrinkage_operator_p if pos else shrinkage_operator
    for _ in range(n_steps):
        w = sh(yk - ss*(a_mtx.dot(yk) - b), beta * ss)
        tk = 0.5 * (1 + np.sqrt(1 + 4*tk_prev**2))
        yk = w + (tk_prev - 1) / tk * (w - yk)
        tk_prev = tk
        n2 = np.linalg.norm(w, ord=2)
        if np.isclose(n1, n2, rtol=0, atol=atol):
            break
        else:
            n1 = n2
    return w

### L2 Regularization optimization ###
def ls_solution(
    x: np.ndarray, 
    y: np.ndarray, 
    beta: float, 
) -> np.ndarray:
    a_mtx, b = x.T.conj().dot(x), x.T.conj().dot(y)
    if beta: 
        a_mtx += beta * np.eye(a_mtx.shape[0])
    return np.linalg.solve(a_mtx, b)

### LS optimization with fixed norm ###
def _nl_mu(x, ss, sc):
    num = (sc)**2
    denom = (ss + 2*x)**2
    return np.sum(num / denom)

def nl_mu(x, ss, sc, alp):
    """ 
    Function to find x(mu) lagrange multiplier optimal value by Newton method. 
    """
    return 1/_nl_mu(x, ss, sc) - 1/alp

def lsc_solution(
    x: np.ndarray, 
    y: np.ndarray, 
    alp: float, 
    mu0: float = 0.1
) -> np.ndarray:
    """
    Solution of Least Squares problem with fixed norm-2 = alp.
    """
    u, s, vt = svd(x, full_matrices=False)
    c = u.T.dot(y)
    ss, sc = s**2, s*c
    mu = newton(nl_mu, mu0, args=(ss, sc, alp), disp=False) # THINK ABOUT THIS ONE 
    return vt.T.dot(sc / (ss + 2*mu))
