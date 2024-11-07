"""
This module contains the KNClassifier class which is a wrapper around.

The KNeighborsClassifier from scikit-learn for classification tasks.
"""

from typing import ClassVar

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from autoop.core.ml.model.model import Model


class KNClassifier(Model):
    """
    A wrapper around the KNeighborsClassifier from.

    scikit-learn for classification tasks.
    """

    parameters: dict = None
    _model: ClassVar[KNeighborsClassifier] = KNeighborsClassifier()
    type: ClassVar[str] = "classification"

    def fit(
        self: "KNClassifier",
        observations: np.ndarray,
        ground_truth: np.ndarray
    ) -> None:
        """
        Fit the model to the given observations and ground truth.

        Args:
            observations: np.ndarray: input data
            ground_truth: np.ndarray: target values
        """
        self._model.fit(observations, ground_truth)
        self.parameters = {
            "effective_metric": self._model.effective_metric_,
        }

    def predict(self: "KNClassifier", observations: np.ndarray) -> np.ndarray:
        """
        Predict the target values for the given observations.

        Args:
            observations: np.ndarray: input data

        Returns:
            np.ndarray: predicted target values
        """
        return self._model.predict(observations)

    def get_parameters(self: "KNClassifier") -> dict:
        """
        Get the parameters of the model.

        Returns:
            dict: the parameters of the model
        """
        return self.parameters
