"""
This module contains the RidgeClassifier class which is a wrapper around.

sklearn's RidgeClassifier.
"""

from typing import ClassVar

import numpy as np
from sklearn.linear_model import RidgeClassifier

from autoop.core.ml.model.model import Model


class RidgeClassifier(Model):
    """A wrapper around sklearn's RidgeClassifier for classification."""

    parameters: dict = None
    _model: ClassVar[RidgeClassifier] = RidgeClassifier()
    type: ClassVar[str] = "classification"

    def fit(
        self: "RidgeClassifier",
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

    def predict(
        self: "RidgeClassifier", observations: np.ndarray
    ) -> np.ndarray:
        """
        Predict the target values for the given observations.

        Args:
            observations: np.ndarray: input data

        Returns:
            np.ndarray: predicted target values
        """
        return self._model.predict(observations)

    def get_parameters(self: "RidgeClassifier") -> dict:
        """
        Get the parameters of the model.

        Returns:
            dict: the parameters of the model
        """
        return self.parameters
