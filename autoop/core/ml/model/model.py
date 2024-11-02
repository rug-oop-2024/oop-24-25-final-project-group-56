
from abc import ABC, abstractmethod
from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
from typing import Literal
from pydantic import BaseModel, Field, PrivateAttr


class Model(ABC, BaseModel):
    """Base class for all models."""
    _parameters: dict = PrivateAttr(default_factory=dict)
    _strict_parameters: dict = PrivateAttr(default_factory=dict)
    _hyperparameters: dict = PrivateAttr(default_factory=dict)

    @abstractmethod
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray):
        pass

    @abstractmethod
    def predict(self, observations: np.ndarray) -> np.array:
        pass

    @abstractmethod
    def get_parameters(self) -> dict:
        pass
    
