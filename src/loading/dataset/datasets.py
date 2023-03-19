"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: project
from .dataset import DataSet


class UnsupervisedDataSet(DataSet):
    def __init__(self, params, inputs, targets=None):
        # Mother Class
        super(UnsupervisedDataSet, self).__init__(params, inputs, targets)

        # Attributes
        if not self._params["lazy_loading"]:
            self._load_dataset()

    def _collect_data_info(self):
        input_tensor = self.__getitem__(0)
        return {
            "spatial_dims": len(input_tensor.shape) - 2,
            "img_size": tuple(input_tensor.shape[2:]),
            "in_channels": input_tensor.shape[1],
            "out_channels": 1
        }

    def __getitem__(self, idx):
        # 2D data
        if self._params["input_dim"] == 2:
            return self._get_data(self._inputs[idx])

        # 3D data
        elif self._params["input_dim"] == 3:
            return self._data_chopper(self._get_data(self._inputs[idx]))


class SupervisedDataSet(DataSet):
    def __init__(self, params, inputs, targets=None):
        # Mother Class
        super(SupervisedDataSet, self).__init__(params, inputs, targets)

        # Attributes
        if not self._params["lazy_loading"]:
            self._load_dataset()

    def _collect_data_info(self):
        input_tensor, target_tensor = self.__getitem__(0)
        return {
            "spatial_dims": len(input_tensor.shape) - 2,
            "img_size": tuple(input_tensor.shape[2:]),
            "in_channels": input_tensor.shape[1],
            "out_channels": target_tensor.shape[1]
        }

    def __getitem__(self, idx):
        # 2D data
        if self._params["input_dim"] == 2:
            return self._get_data(self._inputs[idx]), self._get_data(self._targets[idx])

        # 3D data
        elif self._params["input_dim"] == 3:
            return self._data_chopper(
                self._get_data(self._inputs[idx]),
                self._get_data(self._targets[idx])
            )
