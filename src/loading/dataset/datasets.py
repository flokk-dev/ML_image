"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: project
from .dataset import DataSet


class DataSet2D(DataSet):
    def __init__(self, params, input_paths, target_paths=None):
        # Mother Class
        super(DataSet2D, self).__init__(params, input_paths, target_paths)

        # Attributes
        self._dim = 2

    def __getitem__(self, index):
        input_tensor = self._adjust_shape(self._data_loader(self._input_paths[index]))
        if self._target_paths is None:
            return input_tensor

        # TARGET
        target_tensor = self._adjust_shape(self._data_loader(self._target_paths[index]))
        return input_tensor, target_tensor


class DataSet3D(DataSet):
    def __init__(self, params, input_paths, target_paths=None):
        # Mother Class
        super(DataSet3D, self).__init__(params, input_paths, target_paths)

        # Attributes
        self._dim = 3

        # Components
        self._data_chopper = self._data_choppers[params["dim"]]()

    def __getitem__(self, index):
        input_tensor = self._adjust_shape(self._data_loader(self._input_paths[index]))
        if self._target_paths is None:
            return self._data_chopper(input_tensor)

        # TARGET
        target_tensor = self._adjust_shape(self._data_loader(self._target_paths[index]))
        return self._data_chopper(input_tensor, target_tensor)
