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

    def __init__(self, params, input_paths):
        # Mother Class
        super(DataSet, self).__init__()

        # Attributes
        self._params = params
        self._input_paths = input_paths

        # Components
        self._data_loader = self._data_loaders[params["file_type"]]()
        self._data_chopper = self._data_choppers[params["output_dim"]]()

    @staticmethod
    def _verify_shape(tensor, dim):
        if not dim <= len(tensor.shape) <= dim + 2:
            raise ValueError(f"The tensor's shape isn't valid: {tensor.shape}")

        if torch.sum((torch.Tensor(tuple(tensor.shape)) > 1)) > len(tensor.shape):
            raise ValueError(f"The tensor's shape isn't valid: {tensor.shape}")

        if torch.sum((torch.Tensor(tuple(tensor.shape)) > 5)) > dim:
            raise ValueError(f"The tensor's shape isn't valid: {tensor.shape}")

    @staticmethod
    def _adjust_shape(tensor, dim):
        if len(tensor.shape) == dim+2:
            if tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)

        if len(tensor.shape) == dim+1 and tensor.shape[-1] == min(tensor.shape):
            dims_order = (dim, *(i for i in range(dim)))
            tensor = torch.permute(tensor, dims_order)

        elif len(tensor.shape) == dim:
            tensor = tensor.unsqueeze(0)

        return tensor.unsqueeze(0)

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        return len(self._input_paths)
