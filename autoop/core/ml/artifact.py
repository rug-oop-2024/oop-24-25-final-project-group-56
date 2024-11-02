from pydantic import BaseModel, Field
from typing import Dict, List
import base64


class Artifact(BaseModel):
    name: str = Field(..., description="Name of the artifact")
    version: str = Field(..., description="Version of the asset")
    asset_path: str = Field(..., description="Path to the asset")
    data: bytes = Field(..., description="Binary state data")
    metadata: Dict[str, str] = Field(
        ..., description="Metadata containing experiment and run IDs"
    )
    id: str = Field(..., description="ID of the artifact")
    type: str = Field(..., description="Type of model")
    tags: List[str] = Field(
        ..., description="Tags associated with the artifact"
    )

    def generate_id(self):
        encoded_path = base64.b64encode(
            self.asset_path.encode('utf-8')
        ).decode('utf-8')
        return f"{encoded_path}:{self.version}"
    

    
