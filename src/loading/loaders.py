"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *

# IMPORT: project
from .loader import Loader

from src.loading.data_loader import UnsupervisedDataLoader, SupervisedDataLoader
from src.loading.dataset import UnsupervisedDataSet, SupervisedDataSet


class UnsupervisedLoader(Loader):
    """
    Represents a loader for unsupervised deep learning problem.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour
        _input_paths : Dict[str, List[str]]
            file paths of the input data

    Methods
    ----------
        _extract_paths
            Extracts file paths from a dataset
        _file_depth : int
            Calculates the depth of the file within the dataset
        _generate_data_loaders : Dict[str, UnsupervisedDataLoader]
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
        # Mother Class
        super(UnsupervisedLoader, self).__init__(params)

    def _extract_paths(self, dataset_path: str):
        """
        Extracts file paths from a dataset.

        Parameters
        ----------
            dataset_path : str
                path to the dataset
        """
        file_paths: List[str] = super()._extract_paths(dataset_path)

        nb_paths: int = len(file_paths)
        for idx in range(nb_paths):
            step: str = "train" if idx < int(nb_paths * self._params["valid_coeff"]) else "valid"

            self._input_paths[step].append(file_paths[idx])

    def _generate_data_loaders(self) -> Dict[str, UnsupervisedDataLoader]:
        """
        Generates data loaders using the extracted file paths.

        Returns
        ----------
            Dict[str, UnsupervisedDataLoader]
                data loaders containing training data
        """
        # Dataset
        train_dataset: UnsupervisedDataSet = UnsupervisedDataSet(
            self._params, self._input_paths["train"]
        )
        valid_dataset: UnsupervisedDataSet = UnsupervisedDataSet(
            self._params, self._input_paths["valid"]
        )

        # Data loader
        return {
            "train": UnsupervisedDataLoader(self._params, train_dataset),
            "valid": UnsupervisedDataLoader(self._params, valid_dataset)
        }

    def __call__(self, dataset_path: str) -> Dict[str, UnsupervisedDataLoader]:
        """
        Parameters
        ----------
            dataset_path : str
                path to the dataset

        Returns
        ----------
            Dict[str, UnsupervisedDataLoader]
                data loaders containing training data
        """
        self._input_paths: Dict[str, List[str]] = {"train": list(), "valid": list()}

        self._extract_paths(dataset_path)
        return self._generate_data_loaders()


class SupervisedLoader(Loader):
    """
    Represents a loader for supervised deep learning problem.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour
        _input_paths : Dict[str, List[str]]
            file paths of the input data
        _target_paths : Dict[str, List[str]]
            file paths of the target data

    Methods
    ----------
        _extract_paths
            Extracts file paths from a dataset
        _file_depth : int
            Calculates the depth of the file within the dataset
        _generate_data_loaders : Dict[str, SupervisedDataLoader]
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
        # Mother Class
        super(SupervisedLoader, self).__init__(params)

        # Attributes
        self._target_paths = {"train": list(), "valid": list()}

    def _extract_paths(self, dataset_path: str):
        """
        Extracts file paths from a dataset.

        Parameters
        ----------
            dataset_path : str
                path to the dataset
        """
        file_paths: List[str] = super()._extract_paths(dataset_path)

        nb_paths: int = len(file_paths)
        for idx in range(0, nb_paths, 2):
            step: str = "train" if idx < int(nb_paths * self._params["valid_coeff"]) else "valid"

            self._input_paths[step].append(file_paths[idx])
            self._target_paths[step].append(file_paths[idx+1])

    def _generate_data_loaders(self) -> Dict[str, SupervisedDataLoader]:
        """
        Generates data loaders using the extracted file paths.

        Returns
        ----------
            Dict[str, SupervisedDataLoader]
                data loaders containing training data
        """
        # Dataset
        train_dataset: SupervisedDataSet = SupervisedDataSet(
            self._params, self._input_paths["train"], self._target_paths["train"]
        )
        valid_dataset: SupervisedDataSet = SupervisedDataSet(
            self._params, self._input_paths["valid"], self._target_paths["valid"]
        )

        # Data loader
        return {
            "train": SupervisedDataLoader(self._params, train_dataset),
            "valid": SupervisedDataLoader(self._params, valid_dataset)
        }

    def __call__(self, dataset_path: str) -> Dict[str, SupervisedDataLoader]:
        """
        Parameters
        ----------
            dataset_path : str
                path to the dataset

        Returns
        ----------
            Dict[str, SupervisedDataLoader]
                data loaders containing training data
        """
        self._input_paths: Dict[str, List[str]] = {"train": list(), "valid": list()}
        self._target_paths: Dict[str, List[str]] = {"train": list(), "valid": list()}

        self._extract_paths(dataset_path)
        return self._generate_data_loaders()
