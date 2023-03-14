"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: project
from .dataset import DataSet


class DataSet2D(DataSet):
    def __init__(self, params, inputs, targets=None):
        # Mother Class
        super(DataSet2D, self).__init__(params, inputs, targets)

        # Attributes
        self._dim = 2

        if not self._params["lazy_loading"]:
            self._load_dataset()

    def __getitem__(self, idx):
        # Unsupervised training
        if self._params["training_type"] == "unsupervised":
            return self._get_data(self._inputs[idx])

        # Supervised training
        elif self._params["training_type"] == "supervised":
            return self._get_data(self._inputs[idx]), self._get_data(self._targets[idx])

        # Semi-supervised training
        elif self._params["training_type"] == "semi-supervised":
            raise NotImplementedError()


class DataSet3D(DataSet):
    def __init__(self, params, inputs, targets=None):
        # Mother Class
        super(DataSet3D, self).__init__(params, inputs, targets)

        # Attributes
        self._dim = 3

        if not self._params["lazy_loading"]:
            self._load_dataset()

    def __getitem__(self, idx):
        # Unsupervised training
        if self._params["training_type"] == "unsupervised":
            return self._data_chopper(self._get_data(self._inputs[idx]))

        # Supervised training
        elif self._params["training_type"] == "supervised":
            return self._data_chopper(
                self._get_data(self._inputs[idx]),
                self._get_data(self._targets[idx])
            )

        # Semi-supervised training
        elif self._params["training_type"] == "semi-supervised":
            raise NotImplementedError()
