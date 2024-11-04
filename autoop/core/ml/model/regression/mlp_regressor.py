from typing import ClassVar

import numpy as np
from sklearn.neural_network import MLPRegressor

from autoop.core.ml.model.model import Model


class MLPRegressor(Model):
    """MLP Regressor model."""
    _parameters: dict = None
    _model: ClassVar[MLPRegressor] = MLPRegressor()
    type: ClassVar[str] = "regression"

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """Fit the model to the data.
        Args:
            observations: np.ndarray: input data
            ground_truth: np.ndarray: target data
        """
        self._parameters = {
            None: None
        }
        self._model.fit(observations, ground_truth)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Predict the target variable.
        Args:
            observations: np.ndarray: input data
        Returns:
            np.ndarray: predicted target variable
        """
        return self._model.predict(observations)

    def get_parameters(self) -> dict:
        """Get the parameters of the model.
        Returns:
            dict: parameters of the model
        """
        return self._parameters
