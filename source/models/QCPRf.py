from typing import Optional, Callable
from functools import partial

import numpy as np
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from ..features import Feature, PPFeature, ppf_q2, ff_q2
from ..q_cpr_f import q_cpr_f, predict_score

class QCPRf(RegressorMixin, BaseEstimator):
    """ Quantized CP Regression Model with feature relevance learning (QCPRf). """
    def __init__(
        self, 
        rank: int = 1, 
        fmaps_list: list[Feature] = [PPFeature(),],
        m_order: int = 2,
        init_type: str = 'kj_vec',
        n_epoch: int = 1, 
        alpha: float = 1.0, 
        beta: float = 1.0,
        lambda_reg_type: str = 'l2',
        n_steps_l1: int = 100,
        random_state: Optional[int] = None,
        init_equal_lambda: bool = False,
        positive_lambda: bool = False,
        callback: Optional[Callable] = None,
        update_order_t: str = 'wl' #'lw'
    ):
        _check_input_params(m_order)  
        self.rank = rank
        self.fmaps_list = fmaps_list
        self.m_order = m_order
        self.init_type = init_type
        self.n_epoch = n_epoch
        self.alpha = alpha
        self.beta = beta
        self.lambda_reg_type = lambda_reg_type
        self.n_steps_l1 = n_steps_l1
        self.random_state = random_state
        self.init_equal_lambda = init_equal_lambda
        self.positive_lambda = positive_lambda
        self.callback = callback
        self.update_order_t = update_order_t
        self._dtype = None
    
    def _prepare_feature_mappings(self):
        mappings = []
        for feature in self.fmaps_list:
            if feature.name == 'ppf':
                mappings.append(ppf_q2)
            elif feature.name == 'ff':
                mappings.append(
                    partial(
                        ff_q2, 
                        m_order=self.m_order, 
                        k_d=int(np.log2(self.m_order)), 
                        p_scale=feature.p_scale,
                    )
                )
                self._dtype = np.complex128
            else:
                raise ValueError(f'Bad feature_map name = "{feature.name}". See docs.')
        self._dtype = np.float64 if self._dtype is None else self._dtype
        return mappings

    def fit(self, X, y, xy_test: Optional[tuple] = None):
        X, y = check_X_y(X, y)
        self._feature_maps_list = self._prepare_feature_mappings()
        self.weights_, self.lambdas_, self.kd_ = q_cpr_f(
            X, y, self.m_order, self._feature_maps_list, 
            self.rank, self.init_type, self.n_epoch, self.alpha, 
            self.beta, self.lambda_reg_type, self.n_steps_l1, self.random_state, 
            self.init_equal_lambda, self.positive_lambda, self._dtype,
            xy_test, self.callback, self.update_order_t
        )
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        X = check_array(X)
        check_is_fitted(self, 'is_fitted_')
        return predict_score(
            X, self.kd_, self.weights_, self.lambdas_, self._feature_maps_list)
    
    def score(self, X, y):
        return r2_score(y, self.predict(X))
    
def _check_input_params(m_order: int) -> None:
    if m_order & (m_order - 1):
        raise ValueError(f"m_order should be a power of 2, but it is {m_order}.")
    