from pydantic import BaseModel, Field
from typing import Dict, List
import base64


class Artifact(BaseModel):
    """Artifact class
    args:
        name: name of the artifact
        version: version of the asset
        asset_path: path to the asset
        data: binary state data
        metadata: metadata containing experiment and run IDs
        id: ID of the artifact
        type: type of model
        tags: tags associated with the artifact
    """
    name: str = Field("Artifact", description="Name of the artifact")
    version: str = Field("1.0.0", description="Version of the asset")
    asset_path: str = Field(
        f"assets/dbo/artifacts/{name}",
        description="Path to the asset"
    )
    data: bytes = Field(None, description="Binary state data")
    metadata: Dict[str, str] = Field(
        {}, description="Metadata containing experiment and run IDs"
    )
    id: str = Field("id", description="ID of the artifact")
    type: str = Field("type", description="Type of model")
    tags: List[str] = Field(
        [], description="Tags associated with the artifact"
    )

    def generate_id(self) -> str:
        """Generate ID for the artifact
        returns:
            str: generated ID
        """
        encoded_path = base64.b64encode(
            self.asset_path.encode('utf-8')
        ).decode('utf-8')
        return f"{encoded_path}:{self.version}"
