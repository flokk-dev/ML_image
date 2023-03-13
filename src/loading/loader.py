"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import os

# IMPORT: project
from src.loading.dataset import DataSet2D, DataSet3D


class Loader:
    _datasets = {2: DataSet2D, 3: DataSet3D}

    def __init__(self, params):
        # Attributes
        self._params = params
        self._input_paths = list()

    def _extract_paths(self, path):
        file_paths = list()
        for root, dirs, files in os.walk(path, topdown=False):
            for file_path in map(lambda e: os.path.join(root, e), files):
                if self._file_depth(file_path, self._params["dataset_name"]) == self._params["file_depth"]:
                    file_paths.append(file_path)

        return list(sorted(file_paths))

    @staticmethod
    def _file_depth(path, dataset_name):
        # try to find "dataset_name" in path
        try:
            idx = path.index(f"{os.sep}{dataset_name}{os.sep}")
            return len(path[idx + 1:].split(os.sep)) - 2
        except ValueError:
            # try to find if dataset_name begin the path
            try:
                idx = path.index(f"{dataset_name}{os.sep}")
                return len(path[idx + 1:].split(os.sep)) - 2
            except ValueError:
                raise ValueError(f"\"{dataset_name}\" n'apparait pas dans le chemin spécifié.")

    def _generate_data_loader(self):
        raise NotImplementedError()

    def __call__(self, path):
        self._input_paths = list()
