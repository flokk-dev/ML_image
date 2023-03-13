"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: project
from .loader import Loader
from src.loading.data_loader import UnsupervisedDataLoader, SupervisedDataLoader


class UnsupervisedLoader(Loader):
    def __init__(self, params):
        # Mother Class
        super(UnsupervisedLoader, self).__init__(params)

    def _extract_paths(self, path):
        file_paths = super()._extract_paths(path)

        for idx in range(len(file_paths)):
            self._input_paths.append(file_paths[idx])

    def _generate_data_loader(self):
        dataset = self._datasets[self._params["input_dim"]](
            self._params, self._input_paths
        )
        return UnsupervisedDataLoader(self._params, dataset)

    def __call__(self, path):
        self._extract_paths(path)
        return self._generate_data_loader()


class SupervisedLoader(Loader):
    def __init__(self, params):
        # Mother Class
        super(SupervisedLoader, self).__init__(params)

        # Attributes
        self._target_paths = list()

    def _extract_paths(self, path):
        file_paths = super()._extract_paths(path)

        for idx in range(0, len(file_paths), 2):
            self._input_paths.append(file_paths[idx])
            self._target_paths.append(file_paths[idx+1])

    def _generate_data_loader(self):
        dataset = self._datasets[self._params["input_dim"]](
            self._params, self._input_paths, self._target_paths
        )
        return SupervisedDataLoader(self._params, dataset)

    def __call__(self, path):
        self._target_paths = list()

        self._extract_paths(path)
        return self._generate_data_loader()
