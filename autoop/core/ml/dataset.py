"""This module defines the Dataset class for handling dataset artifacts."""

from autoop.core.ml.artifact import Artifact
import pandas as pd
import io


class Dataset(Artifact):
    """Dataset class for handling dataset artifacts."""

    def __init__(self: "Dataset", *args: object, **kwargs: object) -> None:
        """Initialize the Dataset class."""
        super().__init__(type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(
        data: pd.DataFrame, name: str, asset_path: str, version: str = "1.0.0"
    ) -> "Dataset":
        """Create a dataset from a pandas DataFrame."""
        dataset = Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
            metadata={
                "type": "dataset"
            },
            id="id",
            tags=["dataset"]
        )
        dataset.id = dataset.generate_id()
        return dataset

    def read(self: "Dataset") -> pd.DataFrame:
        """Read the dataset."""
        csv = self.data.decode()
        return pd.read_csv(io.StringIO(csv))

    def save(self: "Dataset", data: pd.DataFrame) -> bytes:
        """Save the dataset."""
        bytes = data.to_csv(index=False).encode()
        return super().save(bytes)
