"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: dataset loading
from .dataset import DataSet


class DataSet2D(DataSet):
    def __init__(self, params: dict, input_paths: list, target_paths: list):
        # Mother Class
        super(DataSet2D, self).__init__(params, input_paths, target_paths)

    def __getitem__(self, index: int) -> tuple:
        return self._data_loader(self._input_paths[index]), \
            self._data_loader(self._target_paths[index])


class DataSet3D(DataSet):
    def __init__(self, params: dict, input_paths: list, target_paths: list):
        # Mother Class
        super(DataSet3D, self).__init__(params, input_paths, target_paths)

        # Components
        self._data_chopper = self._data_choppers[params["dim"]]

    def __getitem__(self, index: int) -> tuple:
        input_tensor = self._data_loader(self._input_paths[index])
        target_tensor = self._data_loader(self._target_paths[index])

        return self._data_chopper(input_tensor, target_tensor)
