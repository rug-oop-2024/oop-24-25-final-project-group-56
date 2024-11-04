from typing import ClassVar

import numpy as np
from sklearn.neural_network import MLPRegressor

from autoop.core.ml.model.model import Model


class MLPRegressor(Model):

    parameters: dict = None
    _model: ClassVar[MLPRegressor] = MLPRegressor()
    type: ClassVar[str] = "regression"
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray):
        self.parameters = {
            None: None
        }
        self._model.fit(observations, ground_truth)

    def predict(self, observations: np.ndarray):
        return self._model.predict(observations)

    def get_parameters(self):
        return self.parameters
