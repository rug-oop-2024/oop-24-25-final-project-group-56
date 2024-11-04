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
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray):
        pass

    @abstractmethod
    def predict(self, observations: np.ndarray) -> np.array:
        pass

    @abstractmethod
    def get_parameters(self) -> dict:
        pass

    def to_artifact(self, name: str) -> Artifact:
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
