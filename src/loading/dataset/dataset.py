"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: dataset loading
from torch.utils.data import Dataset

# IMPORT: project
from .data_loader import ImageLoader, NumpyLoader, ZSTDLoader, TensorLoader
from .data_chopper import DataChopper2D, DataChopper25D, DataChopper3D


class DataSet(Dataset):
    _data_loaders = {"img": ImageLoader, "npy": NumpyLoader, "zstd": ZSTDLoader, "pt": TensorLoader}
    _data_choppers = {2: DataChopper2D, 2.5: DataChopper25D, 3: DataChopper3D}

    def __init__(self, params: dict, input_paths: list, target_paths: list):
        # Mother Class
        super(DataSet, self).__init__()

        # Attributes
        self._params = params

        self._input_paths = input_paths
        self._target_paths = target_paths

        # Components
        self._data_loader = self._data_loaders[params["file_type"]]

    def __getitem__(self, index: int) -> tuple:
        raise NotImplementedError()

    def __len__(self) -> int:
        return len(self._input_paths)
