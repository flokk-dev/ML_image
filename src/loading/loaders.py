"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import typing

# IMPORT: project
from .loader import Loader

from src.loading.data_loader import UnsupervisedDataLoader, SupervisedDataLoader
from src.loading.dataset import UnsupervisedDataSet, SupervisedDataSet


class UnsupervisedLoader(Loader):
    def __init__(
            self,
            params: typing.Dict[str, typing.Any]
    ):
        """
        pass.
        """
        # Mother Class
        super(UnsupervisedLoader, self).__init__(params)

    def _extract_paths(
            self,
            dataset_path: str
    ):
        """
        pass.
        """
        file_paths: typing.List[str] = super()._extract_paths(dataset_path)

        nb_paths: int = len(file_paths)
        for idx in range(nb_paths):
            step: str = "train" if idx < int(nb_paths*0.8) else "valid"
            self._input_paths[step].append(file_paths[idx])

    def _generate_data_loaders(self) -> typing.Dict[str, UnsupervisedDataLoader]:
        """
        pass.
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

    def __call__(
            self,
            dataset_path: str
    ) -> typing.Dict[str, UnsupervisedDataLoader]:
        """
        pass.
        """
        self._input_paths: typing.Dict[str, typing.List[str]] = {"train": list(), "valid": list()}

        self._extract_paths(dataset_path)
        return self._generate_data_loaders()


class SupervisedLoader(Loader):
    def __init__(
            self,
            params: typing.Dict[str, typing.Any]
    ):
        """
        pass.
        """
        # Mother Class
        super(SupervisedLoader, self).__init__(params)

        # Attributes
        self._target_paths = {"train": list(), "valid": list()}

    def _extract_paths(
            self,
            dataset_path: str
    ):
        """
        pass.
        """
        file_paths: typing.List[str] = super()._extract_paths(dataset_path)

        nb_paths: int = len(file_paths)
        for idx in range(0, nb_paths, 2):
            step: str = "train" if idx < int(nb_paths * 0.8) else "valid"

            self._input_paths[step].append(file_paths[idx])
            self._target_paths[step].append(file_paths[idx+1])

    def _generate_data_loaders(self) -> typing.Dict[str, SupervisedDataLoader]:
        """
        pass.
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

    def __call__(
            self,
            dataset_path: str
    ) -> typing.Dict[str, SupervisedDataLoader]:
        """
        pass.
        """
        self._input_paths: typing.Dict[str, typing.List[str]] = {"train": list(), "valid": list()}
        self._target_paths: typing.Dict[str, typing.List[str]] = {"train": list(), "valid": list()}

        self._extract_paths(dataset_path)
        return self._generate_data_loaders()
