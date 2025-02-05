import numpy as np
from scipy.linalg import khatri_rao

def khatri_rao_row(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """ Calculate row-wise Khatri-Rao product of 2 matrices """
    return khatri_rao(a.T, b.T).T
