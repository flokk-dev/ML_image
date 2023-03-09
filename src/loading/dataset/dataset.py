"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: dataset loading
from torch.utils.data import Dataset

# IMPORT: data processing
import torch

# IMPORT: project
from .data_loader import ImageLoader, NumpyLoader, ZSTDLoader, TensorLoader
from .data_chopper import DataChopper2D, DataChopper25D, DataChopper3D


class DataSet(Dataset):
    _data_loaders = {"image": ImageLoader, "numpy": NumpyLoader, "zstd": ZSTDLoader, "tensor": TensorLoader}
    _data_choppers = {2: DataChopper2D, 2.5: DataChopper25D, 3: DataChopper3D}

    def __init__(self, params, inputs):
        # Mother Class
        super(DataSet, self).__init__()

        # Attributes
        self._params = params
        self._inputs = inputs

        # Components
        self._data_loader = self._data_loaders[params["file_type"]]()
        self._data_chopper = self._data_choppers[params["output_dim"]]()

    def _load_data(self, file_paths):
        raise NotImplementedError()

    def _verify_shape(self, tensor):
        if not self._params["input_dim"] <= len(tensor.shape) <= self._params["input_dim"] + 2:
            raise ValueError(f"The tensor's shape isn't valid: {tensor.shape}")

        if len(tensor.shape) > self._params["input_dim"]:
            if torch.sum((torch.Tensor(tuple(tensor.shape)) > 1)) >= len(tensor.shape):
                raise ValueError(f"The tensor's shape isn't valid: {tensor.shape}")

        if torch.sum((torch.Tensor(tuple(tensor.shape)) > 5)) > self._params["input_dim"]:
            raise ValueError(f"The tensor's shape isn't valid: {tensor.shape}")

    def _adjust_shape(self, tensor):
        if len(tensor.shape) == self._params["input_dim"]+2:
            if tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)

        if len(tensor.shape) == self._params["input_dim"]+1 and tensor.shape[-1] == min(tensor.shape):
            dims_order = (self._params["input_dim"], *(i for i in range(self._params["input_dim"])))
            tensor = torch.permute(tensor, dims_order)

        elif len(tensor.shape) == self._params["input_dim"]:
            tensor = tensor.unsqueeze(0)

        return tensor.unsqueeze(0)

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        return len(self._inputs)
