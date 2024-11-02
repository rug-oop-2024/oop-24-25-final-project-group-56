from typing import ClassVar

import numpy as np
from sklearn.linear_model import LinearRegression

from autoop.core.ml.model.model import Model


class MultipleLinearRegression(Model):

    parameters: dict = None
    _model: ClassVar[LinearRegression] = LinearRegression()

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray):
        self._model.fit(observations, ground_truth)
        self.parameters = {
            "coefficients": self._model.coef_,
            "intercept": self._model.intercept_,
        }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        return self._model.predict(observations)

    def get_parameters(self) -> dict:
        return self.parameters
    

