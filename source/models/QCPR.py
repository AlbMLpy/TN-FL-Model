from typing import Optional, Callable
from functools import partial

import numpy as np
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from ..features import Feature, PPFeature, ppf_q2, ff_q2
from ..cpr import cpr
from ..model_functionality import predict_score

class QCPR(RegressorMixin, BaseEstimator):
    """ Quantized CP Regression Model (QCPR). """
    def __init__(
        self, 
        rank: int = 1, 
        feature_map: Feature = PPFeature(), 
        m_order: int = 2,
        init_type: str = 'kj_vec',
        n_epoch: int = 1, 
        alpha: float = 1.0, 
        random_state: Optional[int] = None,
        callback: Optional[Callable] = None,
    ):
        self.rank = rank
        self.feature_map = feature_map
        self.m_order = m_order
        self.init_type = init_type
        self.n_epoch = n_epoch
        self.alpha = alpha
        self.random_state = random_state
        self.callback = callback
        self._dtype = None
        self._quantized = True
    
    def _prepare_feature_mapping(self):
        if self.feature_map.name == 'ppf':
            self._dtype = np.float64
            return ppf_q2
        elif self.feature_map.name == 'ff':
            self._dtype = np.complex128
            return partial(
                ff_q2, 
                m_order=self.m_order, 
                k_d=int(np.log2(self.m_order)), 
                p_scale=self.feature_map.p_scale,
            )
        else:
            raise ValueError(f'Bad feature_map = "{self.feature_map}". See docs.')

    def fit(self, X, y, xy_test: Optional[tuple] = None):
        X, y = check_X_y(X, y)
        self._feature_mapping = self._prepare_feature_mapping()
        self.weights_, self.kd_ = cpr(
            X, y, self._quantized, self.m_order, self._feature_mapping, 
            self.rank, self.init_type, self.n_epoch,
            self.alpha, self.random_state, self._dtype, xy_test, self.callback
        )
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        X = check_array(X)
        check_is_fitted(self, 'is_fitted_')
        return predict_score(X, self.kd_, self.weights_, self._feature_mapping)
    
    def score(self, X, y):
        return r2_score(y, self.predict(X))
