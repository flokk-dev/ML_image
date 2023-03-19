"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: project
from .loader import Loader
from src.loading.data_loader import UnsupervisedDataLoader, SupervisedDataLoader
from src.loading.dataset import UnsupervisedDataSet, SupervisedDataSet


class UnsupervisedLoader(Loader):
    def __init__(self, params):
        # Mother Class
        super(UnsupervisedLoader, self).__init__(params)

    def _extract_paths(self, dataset_path):
        file_paths = super()._extract_paths(dataset_path)

        nb_paths = len(file_paths)
        for idx in range(nb_paths):
            step = "train" if idx < int(nb_paths*0.8) else "valid"
            self._input_paths[step].append(file_paths[idx])

    def _generate_data_loaders(self):
        # Dataset
        train_dataset = UnsupervisedDataSet(self._params, self._input_paths["train"])
        valid_dataset = UnsupervisedDataSet(self._params, self._input_paths["valid"])

        # Data loader
        return {
            "train": UnsupervisedDataLoader(self._params, train_dataset),
            "valid": UnsupervisedDataLoader(self._params, valid_dataset)
        }

    def __call__(self, dataset_path):
        self._input_paths = {"train": list(), "valid": list()}

        self._extract_paths(dataset_path)
        return self._generate_data_loaders()


class SupervisedLoader(Loader):
    def __init__(self, params):
        # Mother Class
        super(SupervisedLoader, self).__init__(params)

        # Attributes
        self._target_paths = {"train": list(), "valid": list()}

    def _extract_paths(self, dataset_path):
        file_paths = super()._extract_paths(dataset_path)

        nb_paths = len(file_paths)
        for idx in range(0, nb_paths, 2):
            step = "train" if idx < int(nb_paths * 0.8) else "valid"

            self._input_paths[step].append(file_paths[idx])
            self._target_paths[step].append(file_paths[idx+1])

    def _generate_data_loaders(self):
        # Dataset
        train_dataset = SupervisedDataSet(
            self._params, self._input_paths["train"], self._target_paths["train"]
        )

        valid_dataset = SupervisedDataSet(
            self._params, self._input_paths["valid"], self._target_paths["valid"]
        )

        # Data loader
        return {
            "train": SupervisedDataLoader(self._params, train_dataset),
            "valid": SupervisedDataLoader(self._params, valid_dataset)
        }

    def __call__(self, dataset_path):
        self._input_paths = {"train": list(), "valid": list()}
        self._target_paths = {"train": list(), "valid": list()}

        self._extract_paths(dataset_path)
        return self._generate_data_loaders()
