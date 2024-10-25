from typing import ClassVar

import numpy as np
from sklearn.linear_model import Lasso

from autoop.core.ml.model.model import Model


class LassoRegression(Model):
    parameters: dict = None
    _model: ClassVar[Lasso] = Lasso()

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray):
        self.parameters = {
            "coefficients": self._model.coef_,
            "intercept": self._model.intercept_
        }
        self._model.fit(observations, ground_truth)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        return self._model.predict(observations)

    def get_parameters(self):
        return self.parameters
