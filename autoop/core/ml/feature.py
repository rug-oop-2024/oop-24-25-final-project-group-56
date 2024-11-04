from pydantic import BaseModel, Field
from typing import Literal


class Feature(BaseModel):
    """Feature class
    args:
        name: name of the feature
        type: type of the feature
    """
    name: str = Field(..., description="Name of the feature")
    type: Literal["categorical", "numerical"] = Field(
        ..., description="Type of the feature"
    )

    def __str__(self) -> str:
        """String representation of the feature
        returns:
            str: string representation of the feature
        """
        return f"{self.name}: {self.type}"
