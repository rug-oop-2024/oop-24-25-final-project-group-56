from typing import ClassVar

import numpy as np
from sklearn.neighbors import KNeighborsClassifier 

from autoop.core.ml.model.model import Model


class KNClassifier(Model):

    parameters: dict = None
    _model: ClassVar[KNeighborsClassifier] = KNeighborsClassifier()
    type: ClassVar[str] = "classification"
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray):
        self._model.fit(observations, ground_truth)
        self.parameters = {
            "effective_metric": self._model.effective_metric_,
        }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        return self._model.predict(observations)

    def get_parameters(self) -> dict:
        return self.parameters
