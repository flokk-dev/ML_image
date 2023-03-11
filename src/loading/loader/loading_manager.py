"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import os

# IMPORT: project
from .data_loader import DataLoader
from src.loading.dataset import DataSet2D, DataSet3D


class Loading:
    _datasets = {2: DataSet2D, 3: DataSet3D}

    def __init__(self, params):
        # Attributes
        self._params = params
        self._input_paths = list()

    def _extract_path(self, path):
        file_paths = list()
        for root, dirs, files in os.walk(path, topdown=False):
            for file_path in map(lambda e: os.path.join(root, e), files):
                if self._dataset_file_depth(file_path, "data") == self._params["file_depth"]:
                    file_paths.append(file_path)

        self._order_paths(list(sorted(file_paths)))

    @staticmethod
    def _dataset_file_depth(path, dataset_name):
        try:
            idx = path.index(f"{os.sep}{dataset_name}{os.sep}")
            return len(path[idx+1:].split(os.sep))

        except ValueError:
            return len(path.split(os.sep))

    def _order_paths(self, file_paths):
        raise NotImplementedError()

    def _generate_dataset(self):
        raise NotImplementedError()

    def _generate_data_loader(self):
        return DataLoader(self._params, self._generate_dataset())

    def __call__(self, path):
        self._input_paths = list()
