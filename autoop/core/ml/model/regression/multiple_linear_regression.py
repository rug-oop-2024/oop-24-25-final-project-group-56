from typing import ClassVar

import numpy as np
from sklearn.linear_model import LinearRegression

from autoop.core.ml.model.model import Model


class MultipleLinearRegression(Model):
    """Multiple linear regression model."""
    _parameters: dict = None
    _model: ClassVar[LinearRegression] = LinearRegression()
    type: ClassVar[str] = "regression"

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """Fit the model to the data.
        Args:
            observations: np.ndarray: input data
            ground_truth: np.ndarray: target data
        """
        self._model.fit(observations, ground_truth)
        self._parameters = {
            "coefficients": self._model.coef_,
            "intercept": self._model.intercept_,
        }

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
