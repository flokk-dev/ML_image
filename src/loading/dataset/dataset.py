"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from tqdm import tqdm

# IMPORT: dataset loading
from torch.utils.data import Dataset

# IMPORT: data processing
import torch

# IMPORT: project
from src.loading.dataset.file_loader import ImageLoader, NpyLoader, NpzLoader, PtLoader
from src.loading.dataset.data_chopper import DataChopper2D, DataChopper25D, DataChopper3D


class DataSet(Dataset):
    _FILE_LOADERS = {"image": ImageLoader, "numpy": NpyLoader, "zstd": NpzLoader, "tensor": PtLoader}
    _DATA_CHOPPERS = {2: DataChopper2D, 2.5: DataChopper25D, 3: DataChopper3D}

    def __init__(self, params, inputs, targets=None):
        # Mother Class
        super(DataSet, self).__init__()

        # Attributes
        self._params = params
        self._dim = None

        self._inputs = inputs
        self._targets = targets

        # Components
        self._file_loader = self._FILE_LOADERS[params["file_type"]]()
        self._data_chopper = self._DATA_CHOPPERS[params["output_dim"]]()

    def _load_dataset(self):
        for idx, file_path in enumerate(tqdm(self._inputs, desc="Loading the data in RAM.")):
            self._inputs[idx] = self._get_data(self._inputs[idx])

            if self._targets is not None:
                self._targets[idx] = self._get_data(self._targets[idx])

    def _get_data(self, tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor

        # LOAD
        tensor = self._file_loader(tensor)

        # VERIFY SHAPE
        self._verify_shape(tensor)

        # ADJUST SHAPE
        return self._adjust_shape(tensor)

    def _verify_shape(self, tensor):
        # IF too much dimensions
        if not self._dim <= len(tensor.shape) <= self._dim + 2:
            raise ValueError(f"The tensor's shape isn't valid: {tensor.shape}")

        # IF not 2d tensor a priori
        if torch.sum((torch.Tensor(tuple(tensor.shape)) > 5)) > self._dim:
            raise ValueError(f"The tensor's shape isn't valid: {tensor.shape}")

        # IF not 2d tensor a priori
        if len(tensor.shape) > self._dim:
            if torch.sum((torch.Tensor(tuple(tensor.shape)) > 1)) >= self._dim + 2:
                raise ValueError(f"The tensor's shape isn't valid: {tensor.shape}")

        # IF valid 2d tensor
        return True

    def _adjust_shape(self, tensor):
        if len(tensor.shape) == self._dim+2:
            if tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)

        if len(tensor.shape) == self._dim+1 and tensor.shape[-1] == min(tensor.shape):
            dims_order = (self._dim, *(i for i in range(self._dim)))
            tensor = torch.permute(tensor, dims_order)

        elif len(tensor.shape) == self._dim:
            tensor = tensor.unsqueeze(0)

        return tensor.unsqueeze(0)

    def __getitem__(self, idx):
        raise NotImplementedError()

    def __len__(self):
        return len(self._inputs)
