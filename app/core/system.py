"""System class for managing artifacts and metadata."""
from autoop.core.storage import LocalStorage
from autoop.core.database import Database
from autoop.core.ml.artifact import Artifact
from autoop.core.storage import Storage
from typing import List


class ArtifactRegistry():
    """System class for managing artifacts and metadata."""

    def __init__(self: "ArtifactRegistry",
                 database: Database,
                 storage: Storage) -> None:
        """Initialize a new artifact registry.

        Args:
            database (Database): The database to use
            storage (Storage): The storage to use
        """
        self._database = database
        self._storage = storage

    def register(self: "ArtifactRegistry", artifact: Artifact) -> None:
        """Register an artifact in the registry.

        Args:
            artifact (Artifact): The artifact to register
        """
        # save the artifact in the storage
        self._storage.save(artifact.data, artifact.asset_path)
        # save the metadata in the database
        entry = {
            "name": artifact.name,
            "version": artifact.version,
            "asset_path": artifact.asset_path,
            "tags": artifact.tags,
            "metadata": artifact.metadata,
            "type": artifact.type,
        }
        self._database.set("artifacts", artifact.name, entry)

    def list(self: "ArtifactRegistry", type: str = None) -> List[Artifact]:
        """List all artifacts in the registry.

        Args:
            type (str): The type of the artifact to list

        Returns:
            List[Artifact]: The list of artifacts
        """
        entries = self._database.list("artifacts")
        artifacts = []
        for id, data in entries:
            if type is not None and data["type"] != type:
                continue
            artifact = Artifact(
                name=data["name"],
                version=data["version"],
                asset_path=data["asset_path"],
                tags=data["tags"],
                metadata=data["metadata"],
                data=self._storage.load(data["asset_path"]),
                type=data["type"],
                id=id
            )
            artifacts.append(artifact)
        return artifacts

    def get(self: "ArtifactRegistry", artifact_id: str) -> Artifact:
        """Get an artifact by its id.

        Args:
            artifact_id (str): The id of the artifact to retrieve

        Returns:
            Artifact: The artifact with the given id
        """
        data = self._database.get("artifacts", artifact_id)
        return Artifact(
            name=data["name"],
            version=data["version"],
            asset_path=data["asset_path"],
            tags=data["tags"],
            metadata=data["metadata"],
            id=artifact_id,
            data=self._storage.load(data["asset_path"]),
            type=data["type"],
        )

    def delete(self: "ArtifactRegistry", artifact_id: str) -> None:
        """Delete an artifact by its id.

        Args:
            artifact_id (str): The id of the artifact to delete
        """
        data = self._database.get("artifacts", artifact_id)
        self._storage.delete(data["asset_path"])
        self._database.delete("artifacts", artifact_id)


class AutoMLSystem:
    """System class for managing artifacts and metadata."""

    _instance = None

    def __init__(self: "AutoMLSystem",
                 storage: LocalStorage,
                 database: Database) -> None:
        """Initialize a new artifact registry.

        Args:
            storage (LocalStorage): The storage to use
            database (Database): The database to use
        """
        self._storage = storage
        self._database = database
        self._registry = ArtifactRegistry(database, storage)

    @staticmethod
    def get_instance() -> "AutoMLSystem":
        """Get the singleton instance of the AutoMLSystem class."""
        if AutoMLSystem._instance is None:
            AutoMLSystem._instance = AutoMLSystem(
                LocalStorage("./assets/objects"),
                Database(
                    LocalStorage("./assets/dbo")
                )
            )
        AutoMLSystem._instance._database.refresh()
        return AutoMLSystem._instance

    @property
    def registry(self: "AutoMLSystem") -> ArtifactRegistry:
        """Get the artifact registry."""
        return self._registry
