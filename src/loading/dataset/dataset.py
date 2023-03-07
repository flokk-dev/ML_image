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

    def __init__(self, params, input_paths, target_paths=None):
        # Mother Class
        super(DataSet, self).__init__()

        # Attributes
        self._params = params
        self._dim = None

        self._input_paths = input_paths
        self._target_paths = target_paths

        # Components
        self._data_loader = self._data_loaders[params["file_type"]]()

    def _adjust_shape(self, tensor):
        if not self._dim <= len(tensor.shape) <= self._dim + 2:
            raise ValueError(f"The tensor's shape isn't valid: {tensor.shape}")

        if torch.sum((torch.Tensor(tuple(tensor.shape)) > 5)) > self._dim:
            raise ValueError(f"The tensor's shape isn't valid: {tensor.shape}")

        if len(tensor.shape) == self._dim+2:
            if tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            else:
                raise ValueError(f"The tensor's shape isn't valid: {tensor.shape}")

        if len(tensor.shape) == self._dim+1 and tensor.shape[-1] == min(tensor.shape):
            dims_order = (self._dim, *(i for i in range(self._dim)))
            tensor = torch.permute(tensor, dims_order)

        elif len(tensor.shape) == self._dim:
            tensor = tensor.unsqueeze(0)

        return tensor.unsqueeze(0)

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        return len(self._input_paths)
