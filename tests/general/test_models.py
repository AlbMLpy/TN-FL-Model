import sys
import unittest
from typing import Callable

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler

sys.path.append('./')

from source.models.QCPR import QCPR
from source.models.QCPRf import QCPRf
from source.features import PPFeature, FFeature
from source.loss import ls_l2w_l2l_loss, ls_loss, l2_reg, ls_l2_loss

def mse_l2wl_loss(
    y: np.ndarray, 
    y_pred: np.ndarray, 
    weights: np.ndarray, 
    lambdas: np.ndarray,
    alpha: float,
    beta: float,
) -> float:
    return ls_loss(y, y_pred) + alpha * l2_reg(weights) + beta * l2_reg(lambdas)

def prepare_callback_no_fl():
    def callback_function(params: dict):
        if not hasattr(callback_function, 'data'):
            callback_function.data = [] 
        value = ls_l2_loss(
            params['y'], params['y_pred'], params['weights'], params['alpha'])
        callback_function.data.append(value)
    return callback_function

def prepare_callback_fl(loss_funct: Callable):
    def callback_function(params: dict):
        if not hasattr(callback_function, 'data'):
            callback_function.data = []   
        value = loss_funct(
            params['y'], params['y_pred'], params['weights'], 
            params['lambdas'], params['alpha'], params['beta'])
        callback_function.data.append(value)
    return callback_function

class TestModels(unittest.TestCase):
    def setUp(self) -> None:
        diabetes = load_diabetes()
        x, y = diabetes.data, diabetes.target
        self.x = MinMaxScaler().fit_transform(x)
        self.y = (y - y.mean()) / y.std()
        self.model_params = dict(rank=8, m_order=4, n_epoch=10, alpha=0.001, random_state=0)

    def test_qcpr_loss(self):
        callback_function = prepare_callback_no_fl()
        model_params = dict(callback=callback_function) | self.model_params
        model = QCPR(**model_params)
        model.fit(self.x, self.y) 

        expected = sorted(callback_function.data, reverse=True)
        actual = callback_function.data
        self.assertTrue(np.allclose(actual, expected))

    def test_qcprf_loss(self):        
        callback_function = prepare_callback_fl(ls_l2w_l2l_loss)
        model_params = (
            self.model_params 
            | dict(callback=callback_function, beta=0.001, 
                fmaps_list=[PPFeature(), FFeature(p_scale=2)])
        ) 
        model = QCPRf(**model_params)
        model.fit(self.x, self.y) 

        expected = sorted(callback_function.data, reverse=True)
        actual = callback_function.data
        self.assertTrue(np.allclose(actual, expected))
