import sys
import unittest

import numpy as np

sys.path.append('./')

from source.features import (
    ff_q2,
    ppf_q2,
    fourier_features,
    pure_poli_features, 
    gaussian_kernel_features,
)
from source.matrix_operations import khatri_rao_row

class TestFeatures(unittest.TestCase):
    def test_pure_poli_features(self):
        expected = np.array(
            [
                [1, 0, 0, 0],
                [1, 1, 1, 1],
                [1, 2, 4, 8],
            ]
        )
        actual = pure_poli_features(np.arange(3), None, 4)
        self.assertTrue(np.allclose(actual, expected))
    
    def test_gaussian_kernel_features(self):
        expected = np.array(
            [
                [ 0.18846226,  0.05777485],
                [-0.77123494, -0.10430503],
                [-0.52870649,  0.13053439],
            ]
        )
        actual = gaussian_kernel_features(
            np.array([np.pi, 2*np.pi, 3*np.pi]),
            None, 
            order=2,
            lscale=1,
            domain_bound=1,
        )
        self.assertTrue(np.allclose(actual, expected))

    def test_q2_poli_features(self):
        expected = np.array(
            [
                [1, 0],
                [1, 1],
                [1, 16],
                [1, 81]
            ]
        )
        actual = ppf_q2(
            np.arange(4), 
            q=2,
        )
        self.assertTrue(np.allclose(actual, expected))

    def test_q2_fourier_features(self):
        expected = np.array(
            [
                [ 1.+0.0000000e+00j,  1.+0.0000000e+00j],
                [-1.-1.2246468e-16j, -1.+1.2246468e-16j],
                [ 1.+2.4492936e-16j,  1.-2.4492936e-16j],
                [-1.-3.6739404e-16j, -1.+3.6739404e-16j],
            ]
        )
        actual = ff_q2(
            np.arange(4), 
            q=1,
            m_order=3,
            k_d=3,
            p_scale=1
        )
        self.assertTrue(np.allclose(actual, expected))

    def test_q_nq_poli_features(self):
        x = np.array([1, 2, 3])
        m_order = 256
        k_d = int(np.log2(m_order))

        expected = pure_poli_features(x, None, m_order)
        actual = ppf_q2(x, 0)
        for q in range(1, k_d):
            actual = khatri_rao_row(ppf_q2(x, q), actual)
        self.assertTrue(np.allclose(actual, expected))

    def test_q_nq_fourier_features(self):
        x = np.array([0.1, 0.2, 1, 20])
        m_order = 2**8
        p_scale = 20
        k_d = int(np.log2(m_order))

        actual = ff_q2(x, 0, m_order, k_d, p_scale)
        for q in range(1, k_d):
            actual = khatri_rao_row(ff_q2(x, q, m_order, k_d, p_scale), actual)

        expected = fourier_features(x, None, m_order, p_scale)
        self.assertTrue(np.allclose(actual, expected))
