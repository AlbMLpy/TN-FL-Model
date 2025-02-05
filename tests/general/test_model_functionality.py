import sys
import unittest
from functools import partial

import numpy as np

sys.path.append('./')

from source.model_functionality import (
    init_weights, 
    get_fw_hadamard_mtx,
    get_ww_hadamard_mtx,
    get_updated_als_factor,
    weights3d_to_quantized4d,
    weights4d_quantized_to_3d,
)
from source.features import pure_poli_features, ppf_q2


class TestModelFunctionality(unittest.TestCase):
    def test_init_weights_non_quant(self):
        m_order, rank, d_dim, q_base = 13, 5, 4, None
        temp, _ = init_weights(m_order, rank, d_dim, q_base)
        
        expected = np.array([d_dim, m_order, rank])
        actual = temp.shape
        self.assertTrue(np.allclose(actual, expected))
    
    def test_init_weights_quant(self):
        m_order, rank, d_dim, q_base = 16, 5, 4, 2
        temp, _ = init_weights(m_order, rank, d_dim, q_base)
        
        expected = np.array(
            [d_dim*int(np.emath.logn(q_base, m_order)), q_base, rank])
        actual = temp.shape
        self.assertTrue(np.allclose(actual, expected))

    def test_init_weights_bad_m_order(self):
        # Quantized setting:
        m_order, rank, d_dim, q_base = 13, 5, 4, 2
        with self.assertRaises(ValueError):
            init_weights(m_order, rank, d_dim, q_base)

    def test_get_updated_als_factor(self):
        n, f_dim = 3, 2
        fk_mtx = np.ones((n, f_dim))
        fw_hadamard = np.array(
             [[1.0, 2], [2, 4], [4, 8]]
        )
        ww_hadamard = np.array([[1.0, 3], [3, 5]])
        y = np.array([1.0, 0, 1])
        alpha = 1.0
        
        expected = np.array(
            [
                [0.03846154, 0.03846154],
                [0.03846154, 0.03846154]
            ]
        )
        actual = get_updated_als_factor(
            fk_mtx, 
            fw_hadamard,
            ww_hadamard, 
            y, 
            alpha,
        )
        self.assertTrue(np.allclose(actual, expected))

    def test_get_fw_hadamard_mtx_quant(self):
        x = np.array(
            [
                [1, 1],
                [2, 2],
                [3, 3],
                [4, 4],
            ]
        )
        k_d = 2
        weights = np.array(
            [
                [[1, 2], [2, 3]], 
                [[0, 1], [1, 0]],
                [[1, 2], [2, 3]], 
                [[0, 1], [1, 0]]
            ]
        )
        feature_map = ppf_q2

        expected = np.array(
            [
                [9., 25],
                [400., 64.],
                [3969., 121.],
                [20736., 196.]
            ]
        )
        actual = get_fw_hadamard_mtx(x, k_d, weights, feature_map)
        self.assertTrue(np.allclose(actual, expected))

    def test_get_fw_hadamard_mtx_non_quant(self):
        x = np.array(
            [
                [1, 1],
                [2, 2],
                [3, 3],
                [4, 4],
            ]
        )
        k_d = 1
        weights = np.array(
            [
                [[1, 2], [2, 3], [3, 4]], 
                [[0, 1], [1, 0], [1, 1]]
            ]
        )
        feature_map = partial(pure_poli_features, order=3)

        expected = np.array(
            [
                [  12.,   18.],
                [ 102.,  120.],
                [ 408.,  470.],
                [1140., 1326.]
            ]
        )
        actual = get_fw_hadamard_mtx(x, k_d, weights, feature_map)
        self.assertTrue(np.allclose(actual, expected))

    def test_get_ww_hadamard_mtx(self):
        weights = np.array(
            [
                [[1, 2], [2, 3], [3, 4]], 
                [[0, 1], [1, 0], [1, 1]]
            ]
        )

        expected = np.array([[28., 20.], [20., 58.]])
        actual = get_ww_hadamard_mtx(weights)
        self.assertTrue(np.allclose(actual, expected))

    def test_weights_transformations(self):
        m_order, rank, d_dim, q_base = 4, 2, 3, 2
        w_exp, kd_exp = init_weights(m_order, rank, d_dim, q_base, seed=13)
        w_act, kd_act = weights4d_quantized_to_3d(weights3d_to_quantized4d(w_exp, kd_exp))
        self.assertTrue(np.allclose(w_act, w_exp))
        self.assertEqual(kd_act, kd_exp)
