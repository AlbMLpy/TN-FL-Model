from typing import Callable
from collections import namedtuple

import numpy as np

Feature = tuple
PPFeature = namedtuple('PPFeature', 'name', defaults=['ppf'])
FFeature = namedtuple('FFeature', 'p_scale, name', defaults=[1, 'ff'])
RBFFeature = namedtuple('RBFFeature', 'l_scale, name', defaults=[1, 'rbff'])
FeatureMap = Callable[..., np.ndarray]

def pure_poli_features(
    x: np.ndarray, 
    q: int, # Dummy
    order: int
) -> np.ndarray:
    """ 
    Pure polinomial features matrix for x. 

    References: "Quantized Fourier and Polynomial Features for more 
        Expressive Tensor Network Models", Frederiek Wesel, Kim Batselier, (Definition 3.1).
    """
    return np.power(x[:, None], np.arange(order))

def ppf_q2(x: np.ndarray, q: int) -> np.ndarray:
    """ 
    Quantized pure polinomial features matrix for x. 
    
    References: "Quantized Fourier and Polynomial Features for more 
        Expressive Tensor Network Models", Frederiek Wesel, Kim Batselier, (Definition 3.4).

    NOTE: q should start with 0 -> [log2(m_order) - 1] including
    """
    return np.power(x[:, None], [0, 2**q])

def gaussian_kernel_features(
    x: np.ndarray,
    q: int,  # Dummy
    order: int, 
    lscale: float = 1, 
    domain_bound: float = 1,
) -> np.ndarray:
    """ 
    Gaussian (squared exp.) kernel features matrix for x. 

    References: "Hilbert Space Methods for Reduced-Rank Gaussian Process Regression", 
        Simo Särkkä, (formulas 56, 68(d=1, s=1)).
    """
    x = (x + domain_bound)
    w_scaled = np.pi * np.arange(1, order + 1) / (2 * domain_bound)
    sd = np.sqrt(2 * np.pi) * lscale * np.exp(-np.power(lscale * w_scaled, 2) / 2)
    return np.sqrt(sd / domain_bound) * np.sin(np.outer(x, w_scaled)) 

def fourier_features(
    x: np.ndarray,
    q: int, # Dummy
    m_order: int, 
    p_scale: float = 1, 
):
    """ 
    Fourier Features matrix for x. 

    References: 
        - "Learning multidimensional Fourier series with tensor trains",
            Sander Wahls, Visa Koivunen, H Vincent Poor, Michel Verhaegen.
        - "Quantized Fourier and Polynomial Features for more 
            Expressive Tensor Network Models", Frederiek Wesel, Kim Batselier, (Definition 3.2).
    """
    w = np.arange(-m_order / 2, m_order / 2)
    return np.exp(1j * 2 * np.pi * np.outer(x, w) / p_scale)

def ff_q2(
    x: np.ndarray, 
    q: int, 
    m_order: int, 
    k_d: int, 
    p_scale: float = 1
) -> np.ndarray:
    """ 
    Quantized Fourier Features matrix for x. 

    References: 
        - "Learning multidimensional Fourier series with tensor trains",
            Sander Wahls, Visa Koivunen, H Vincent Poor, Michel Verhaegen.
        - "Quantized Fourier and Polynomial Features for more 
            Expressive Tensor Network Models", Frederiek Wesel, Kim Batselier, (Corollary 3.6).

    NOTE: q should start with 0 -> [log2(m_order) - 1] including
    """
    return np.hstack(
        (
            np.exp(-1j * np.pi * x * m_order / (k_d * p_scale))[:, None], 
            np.exp(1j * np.pi * (-x * m_order / k_d + 2*x*(2**(q))) / p_scale)[:, None]
        ),
    )
