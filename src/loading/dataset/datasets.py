"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: project
from .dataset import DataSet


class DataSetUnsupervised(DataSet):
    def __init__(self, params, inputs):
        # Mother Class
        super(DataSetUnsupervised, self).__init__(params, inputs)

    def __getitem__(self, index):
        # LOAD
        input_tensor = self._data_loader(self._inputs[index])

        # VERIFY SHAPE
        self._verify_shape(input_tensor)

        # ADJUST SHAPE
        input_tensor = self._adjust_shape(input_tensor)

        if self._params["input_dim"] == 2:
            return input_tensor
        return self._data_chopper(input_tensor)


class DataSetSupervised(DataSet):
    def __init__(self, params, input_paths, targets):
        # Mother Class
        super(DataSetSupervised, self).__init__(params, input_paths)

        # Attributes
        self._targets = targets

    def __getitem__(self, index):
        # LOAD
        input_tensor = self._data_loader(self._inputs[index])
        target_tensor = self._data_loader(self._targets[index])

        # VERIFY SHAPE
        self._verify_shape(input_tensor)
        self._verify_shape(target_tensor)

        # ADJUST SHAPE
        input_tensor = self._adjust_shape(input_tensor)
        target_tensor = self._adjust_shape(target_tensor)

        if self._params["input_dim"] == 2:
            return input_tensor, target_tensor
        return self._data_chopper(input_tensor, target_tensor)
