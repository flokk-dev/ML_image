"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import os

# IMPORT: project
from .loading_manager import LoadingManager

from src.loading.dataset import DataSetUnsupervised, DataSetSupervised
from .data_loader import LazyLoader, TensorLoader


class LoadingManagerUnsupervised(LoadingManager):
    def __init__(self, params):
        # Mother Class
        super(LoadingManagerUnsupervised, self).__init__(params)

        # Attributes
        self._dataset = DataSetUnsupervised

    def _order_paths(self, file_paths):
        for idx in range(len(file_paths)):
            self._input_paths.append(file_paths[idx])

    def _generate_data_loader(self):
        dataset = self._dataset(self._params, self._input_paths)

        if self._params["lazy_loading"]:
            return LazyLoader(self._params, dataset)
        return TensorLoader(self._params, dataset)

    def __call__(self, path):
        super().__call__(path)

        self._extract_path(path)
        return self._generate_data_loader()


class LoadingManagerSupervised(LoadingManager):
    def __init__(self, params):
        # Mother Class
        super(LoadingManagerSupervised, self).__init__(params)

        # Attributes
        self._dataset = DataSetSupervised
        self._target_paths = list()

    def _order_paths(self, file_paths):
        for idx in range(0, len(file_paths), 2):
            self._input_paths.append(file_paths[idx])
            self._target_paths.append(file_paths[idx+1])

    def _generate_data_loader(self):
        dataset = self._dataset(self._params, self._input_paths, self._target_paths)

        if self._params["lazy_loading"]:
            return LazyLoader(self._params, dataset)
        return TensorLoader(self._params, dataset)

    def __call__(self, path):
        super().__call__(path)
        self._target_paths = list()

        self._extract_path(path)
        return self._generate_data_loader()
