from typing import ClassVar

import numpy as np
from sklearn.neural_network import MLPRegressor

from autoop.core.ml.model.model import Model


class MLPRegressor(Model):

    _parameters: dict = None
    _model: ClassVar[MLPRegressor] = MLPRegressor()

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray):
        self._parameters = {
            "coefficients": self._model.coefs_,
            "intercept": self._model.intercepts_,
        }
        self._model.fit(observations, ground_truth)

    def predict(self, observations: np.ndarray):
        return self._model.predict(observations)

    def get_parameters(self):
        return self._parameters
