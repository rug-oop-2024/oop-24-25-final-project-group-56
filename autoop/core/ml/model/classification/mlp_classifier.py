from typing import ClassVar

import numpy as np
from sklearn.neural_network import MLPClassifier

from autoop.core.ml.model.model import Model


class MLPClassifier(Model):
    parameters: dict = None
    _model: ClassVar[MLPClassifier] = MLPClassifier()
    type: ClassVar[str] = "classification"
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray):
        self._model.fit(observations, ground_truth)
        self.parameters = {
            "coeficients": self._model.coefs_,
            "intercept": self._model.intercepts_,
        }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        return self._model.predict(observations)

    def get_parameters(self) -> dict:
        return self.parameters
