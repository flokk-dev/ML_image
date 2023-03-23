"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *
import os

# IMPORT: project
from .data_loader import DataLoader


class Loader:
    """
    Represents a general loader, that will be derived depending on the use case.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour
        _input_paths : Dict[str, List[str]]
            the file paths of the input data

    Methods
    ----------
        _extract_paths : List[str]
            Extracts file paths from a dataset
        _file_depth : int
            Calculates the depth of the file within the dataset
        _generate_data_loaders : Dict[str, DataLoader]
            Verifies the tensor's shape according to the desired dimension
    """

    def __init__(self, params: Dict[str, Any]):
        """
        Instantiates a Loader.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour
        """
        # Attributes
        self._params: Dict[str, Any] = params
        self._input_paths: Dict[str, List[str]] = {"train": list(), "valid": list()}

    def _extract_paths(self, dataset_path: str) -> List[str]:
        """
        Extracts file paths from a dataset.

        Parameters
        ----------
            dataset_path : str
                path to the dataset

        Returns
        ----------
            List[str]
                file paths within the dataset
        """
        file_paths: List[str] = list()

        for root, dirs, files in os.walk(dataset_path, topdown=False):
            for file_path in map(lambda e: os.path.join(root, e), files):
                if self._file_depth(file_path, self._params["dataset_name"]) == self._params["file_depth"]:
                    file_paths.append(file_path)

        return list(sorted(file_paths))

    @staticmethod
    def _file_depth(path: str, dataset_name: str) -> int:
        """
        Calculates the depth of the file within the dataset

        Parameters
        ----------
            path : str
                file path whose depth is to be calculated
            dataset_name: str
                name of the dataset

        Returns
        ----------
            int
                depth of the file within the dataset
        """
        # try to find "dataset_name" in path
        try:
            idx: int = path.index(f"{os.sep}{dataset_name}{os.sep}")
            return len(path[idx + 1:].split(os.sep)) - 2
        except ValueError:
            # try to find if dataset_name begin the path
            try:
                idx: int = path.index(f"{dataset_name}{os.sep}")
                return len(path[idx + 1:].split(os.sep)) - 2
            except ValueError:
                raise ValueError(f"\"{dataset_name}\" n'apparait pas dans le chemin spécifié.")

    def _generate_data_loaders(self) -> Dict[str, DataLoader]:
        """
        Generates data loaders using the extracted file paths.

        Returns
        ----------
            Dict[str, DataLoader]
                the data loaders containing training data
        """
        raise NotImplementedError()

    def __call__(self, dataset_path: str) -> Dict[str, DataLoader]:
        """
        Parameters
        ----------
            dataset_path : str
                path to the dataset

        Raises
        ----------
            NotImplementedError
                function isn't implemented yet
        """
        raise NotImplementedError()
