"""
This module contains the MLPRegressor class which is a wrapper.

The MLPRegressor wraps around the MLPRegressor model from scikit-learn.
"""

from typing import ClassVar

import numpy as np
from sklearn.neural_network import MLPRegressor

from autoop.core.ml.model.model import Model


class MLPRegressor(Model):
    """MLP Regressor model."""

    parameters: dict = None
    _model: ClassVar[MLPRegressor] = MLPRegressor()
    type: ClassVar[str] = "regression"

    def fit(
        self: "MLPRegressor",
        observations: np.ndarray,
        ground_truth: np.ndarray
    ) -> None:
        """Fit the model to the data.

        Args:
            observations: np.ndarray: input data
            ground_truth: np.ndarray: target data

        Returns:
            None
        """
        self._model.fit(observations, ground_truth)
        self._parameters = {
            "hidden_layer_sizes": self._model.hidden_layer_sizes,
            "activation": self._model.activation,
            "solver": self._model.solver,
            "alpha": self._model.alpha,
        }

    def predict(self: "MLPRegressor", observations: np.ndarray) -> np.ndarray:
        """Predict the target variable.

        Args:
            observations: np.ndarray: input data

        Returns:
            np.ndarray: predicted target variable
        """
        return self._model.predict(observations)

    def get_parameters(self: "MLPRegressor") -> dict:
        """Get the parameters of the model.

        Returns:
            dict: parameters of the model
        """
        return self._parameters
