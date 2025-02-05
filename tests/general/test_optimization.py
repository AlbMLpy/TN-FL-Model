import sys
import unittest

import numpy as np

sys.path.append('./')

from source.optimization import ls_solution, lsc_solution, fista


class TestOptimization(unittest.TestCase):
    def setUp(self) -> None:
        rs = np.random.RandomState(13)
        self.a_mtx = rs.randn(100, 6)
        self.x_gt = rs.randn(6)
        self.y = self.a_mtx.dot(self.x_gt)
        self.w0 = rs.randn(*self.x_gt.shape)

    def test_fista(self):
        expected = self.x_gt
        actual = fista(self.a_mtx, self.y, self.w0, beta=0, n_steps=100, atol=1e-5)
        self.assertTrue(np.allclose(actual, expected))

    def test_ls_solution(self):        
        expected = self.x_gt
        actual = ls_solution(self.a_mtx, self.y, beta=0)
        self.assertTrue(np.allclose(actual, expected))

    def test_lsc_solution(self):  
        alp = 1      
        w = lsc_solution(self.a_mtx, self.y, alp=alp)

        expected = alp
        actual = np.linalg.norm(w, ord=2) 
        self.assertTrue(np.allclose(actual, expected))
