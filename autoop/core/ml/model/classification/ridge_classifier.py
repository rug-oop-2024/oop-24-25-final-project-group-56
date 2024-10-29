from typing import ClassVar

import numpy as np
from sklearn.linear_model import RidgeClassifier

from autoop.core.ml.model.model import Model


class RidgeClassifier(Model):
    parameters: dict = None
    _model: ClassVar[RidgeClassifier] = RidgeClassifier

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray):
        self.parameters = {
            "coefficients": self._model.coefs_,
            "intercept": self._model.intercepts_,
        }
        self._model.fit(observations, ground_truth)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        return self._model.predict(observations)

    def get_parameters(self) -> dict:
        return self.parameters