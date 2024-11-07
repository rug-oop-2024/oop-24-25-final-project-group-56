"""This module provides a Storage class for managing collections of data."""
from abc import ABC, abstractmethod
import os
from typing import List
from glob import glob


class NotFoundError(Exception):
    """Exception raised when a path is not found."""

    def __init__(self: "NotFoundError", path: str) -> None:
        """Initialize a new NotFoundError.

        Args:
            path (str): The path that was not found
        """
        super().__init__(f"Path not found: {path}")


class Storage(ABC):
    """Storage class for managing collections of data."""

    @abstractmethod
    def save(self: "Storage", data: bytes, path: str) -> None:
        """
        Save data to a given path.

        Args:
            data (bytes): Data to save
            path (str): Path to save data
        """
        pass

    @abstractmethod
    def load(self: "Storage", path: str) -> bytes:
        """
        Load data from a given path.

        Args:
            path (str): Path to load data
        Returns:
            bytes: Loaded data
        """
        pass

    @abstractmethod
    def delete(self: "Storage", path: str) -> None:
        """
        Delete data at a given path.

        Args:
            path (str): Path to delete data
        """
        pass

    @abstractmethod
    def list(self: "Storage", path: str) -> list:
        """
        List all paths under a given path.

        Args:
            path (str): Path to list
        Returns:
            list: List of paths
        """
        pass


class LocalStorage(Storage):
    """Local storage class for managing collections of data."""

    def __init__(self: "LocalStorage", base_path: str = "./assets") -> None:
        """Initialize a new LocalStorage.

        Args:
            base_path (str, optional): The base path to store data
        """
        self._base_path = base_path
        if not os.path.exists(self._base_path):
            os.makedirs(self._base_path)

    def save(self: "LocalStorage", data: bytes, key: str) -> None:
        """Save data to a given path."""
        path = self._join_path(key)
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(data)

    def load(self: "LocalStorage", key: str) -> bytes:
        """Load data from a given path."""
        path = self._join_path(key)
        self._assert_path_exists(path)
        with open(path, 'rb') as f:
            return f.read()

    def delete(self: "LocalStorage", key: str = "/") -> None:
        """Delete data at a given path."""
        self._assert_path_exists(self._join_path(key))
        path = self._join_path(key)
        os.remove(path)

    def list(self: "LocalStorage", prefix: str) -> List[str]:
        """List all paths under a given path."""
        path = self._join_path(prefix)
        self._assert_path_exists(path)
        keys = glob(path + "/**/*", recursive=True)
        return list(filter(os.path.isfile, keys))

    def _assert_path_exists(self: "LocalStorage", path: str) -> None:
        if not os.path.exists(path):
            raise NotFoundError(path)

    def _join_path(self: "LocalStorage", path: str) -> str:
        return os.path.join(self._base_path, path)
