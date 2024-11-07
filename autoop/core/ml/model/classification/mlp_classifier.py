"""
This module contains the MLPClassifier class which is a wrapper.

The wrapped class is sklearn.neural_network.MLPClassifier for classification.
"""

from typing import ClassVar

import numpy as np
from sklearn.neural_network import MLPClassifier

from autoop.core.ml.model.model import Model


class MLPClassifier(Model):
    """A wrapper around MLPClassifier from scikit-learn for classification."""

    parameters: dict = None
    _model: ClassVar[MLPClassifier] = MLPClassifier()
    type: ClassVar[str] = "classification"

    def fit(
        self: "MLPClassifier",
        observations: np.ndarray,
        ground_truth: np.ndarray
    ) -> None:
        """
        Fit the model to the given observations and ground truth.

        Args:
            observations: np.ndarray: input data
            ground_truth: np.ndarray: target values

        Returns:
            None
        """
        self._model.fit(observations, ground_truth)
        self.parameters = {
            "coeficients": self._model.coefs_,
            "intercept": self._model.intercepts_,
        }

    def predict(self: "MLPClassifier", observations: np.ndarray) -> np.ndarray:
        """
        Predict the target values for the given observations.

        Args:
            observations: np.ndarray: input data

        Returns:
            np.ndarray: predicted target values
        """
        return self._model.predict(observations)

    def get_parameters(self: "MLPClassifier") -> dict:
        """
        Get the parameters of the model.

        Returns:
            dict: the parameters of the model
        """
        return self.parameters
