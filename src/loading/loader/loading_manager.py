"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import os


class LoadingManager:
    def __init__(self, params):
        # Attributes
        self._params = params
        self._dataset = None

        self._input_paths = list()

    def _extract_path(self, path):
        file_paths = list()
        for root, dirs, files in os.walk(path, topdown=False):
            for file_path in map(lambda e: os.path.join(root, e), files):
                if self._dataset_file_depth(file_path, "data") == self._params["file_depth"]:
                    file_paths.append(file_path)

        self._order_paths(list(sorted(file_paths)))

    def _order_paths(self, file_paths):
        raise NotImplementedError()

    def _generate_data_loader(self):
        raise NotImplementedError()

    def __call__(self, path):
        self._input_paths = list()

    @staticmethod
    def _dataset_file_depth(path, dataset_name):
        try:
            idx = path.index(f"{os.sep}{dataset_name}{os.sep}")
            return len(path[idx+1:].split(os.sep))

        except ValueError:
            return len(path.split(os.sep))
