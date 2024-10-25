from typing import ClassVar

import numpy as np
from sklearn.linear_model import LogisticRegression

from autoop.core.ml.model.model import Model


class LogisticRegression(Model):

    _parameters: dict = None
    _model: ClassVar[LogisticRegression] = LogisticRegression()

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray):
        self._parameters = {
            "coefficients": self._model.coef_,
            "intercept": self._model.intercept_,
        }
        self._model.fit(observations, ground_truth)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        return self._model.predict(observations)

    def get_parameters(self) -> dict:
        return self._parameters
