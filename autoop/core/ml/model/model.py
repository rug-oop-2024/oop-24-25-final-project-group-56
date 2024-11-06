"""This module contains the base class for all models used in the project."""

from abc import ABC, abstractmethod
from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
from typing import Literal
from pydantic import BaseModel, Field, PrivateAttr


class Model(ABC, BaseModel):
    """Base class for all models.

    abstract methods:
    - fit: fit the model to the data
    - predict: predict the target variable
    - get_parameters: get the parameters of the model
    to_artifact: convert the model to an artifact
    """

    _parameters: dict = PrivateAttr(default_factory=dict)
    _strict_parameters: dict = PrivateAttr(default_factory=dict)
    _hyperparameters: dict = PrivateAttr(default_factory=dict)
    type: Literal["regression", "classification"] = Field(
        ..., description="Type of the model"
    )

    @abstractmethod
    def fit(
        self: "Model", observations: np.ndarray, ground_truth: np.ndarray
    ) -> None:
        """Fit the model to the data.

        Args:
            observations: the input data
            ground_truth: the target variable
        """
        pass

    @abstractmethod
    def predict(self: "Model", observations: np.ndarray) -> np.array:
        """Predict the target variable.

        Args:
            observations: the input data

        Returns:
            np.array: the predicted target variable
        """
        pass

    @abstractmethod
    def get_parameters(self: "Model") -> dict:
        """Get the parameters of the model.

        Returns:
            dict: the parameters of the model
        """
        pass

    def to_artifact(self: "Model", name: str) -> Artifact:
        """Convert the model to an artifact.

        args:
            name: name of the artifact
        returns:
            Artifact: the model as an artifact
        """
        return Artifact(
            name=name,
            type=self.type,
            parameters=deepcopy(self._parameters),
            strict_parameters=deepcopy(self._strict_parameters),
            hyperparameters=deepcopy(self._hyperparameters),
        )
