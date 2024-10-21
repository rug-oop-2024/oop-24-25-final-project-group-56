
from pydantic import BaseModel, Field
from typing import Literal
import numpy as np

from autoop.core.ml.dataset import Dataset


class Feature(BaseModel):
    name: str = Field(..., description="Name of the feature")
    type: Literal["categorical", "numerical"] = Field(
        ..., description="Type of the feature"
    )

    def __str__(self):
        return f"{self.name}: {self.type}"
