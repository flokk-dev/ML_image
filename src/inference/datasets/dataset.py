"""
Creator: HOCQUET Florian, Landza HOUDI
Date: 30/09/2022
Version: 1.0

Purpose: Articualtes the dataset's paths.
"""

# IMPORT: utils
import os
import zstd

# IMPORT: data process
import numpy as np

# IMPORT: deep learning
import torch
from torch.utils.data import Dataset


class DataSet(Dataset):
    """
    The class representing a dataset.

    Attributes:
    - _dataset_info: the dataset information.
    - _inputs: the input datas' path.
    - _targets: the target datas' path.
    - _transform: some transformations to apply to data.

    Methods:
    - _load_volume: Loads volume from path.
    - _my_collate: Defines the way to process data.
    """

    def __init__(self, params: dict, data_info: dict, info: dict):
        """
        Initialize the DataSet class.

        Parameters:
        - data_info (dict): the ditionary needed to load the data.
        - input_paths (list): the input data's paths.
        - target_paths (list): the target data's paths.
        """
        self._params = params
        self._data_info = data_info

        self._inputs = info["inputs"]
        self._targets = info["targets"]
        self._IMC = info["IMC"]

    def __getitem__(self, index: int) -> tuple:
        shape = self._data_info[int(self._inputs[index].split(os.sep)[-3])]["shape"]

        # LOADING
        input_volume = self._load_volume(self._inputs[index], shape)
        target_volume = self._load_volume(self._targets[index], shape)

        return *self._my_collate(input_volume, target_volume), self._IMC[index]

    def _load_volume(self, path: str, shape: list) -> torch.Tensor:
        """
        Loads volume from path.

        Parameters:
        - path (str): the path of the volume to load.
        - shape (list): the shape of the volume.

        Returns:
        - (torch.Tensor): the output tensor.
        """
        decompressed_volume = zstd.decompress(np.load(path))
        flat_volume = np.frombuffer(decompressed_volume, dtype=np.float32).copy()
        volume = np.reshape(flat_volume, shape)

        return torch.from_numpy(volume)

    def _my_collate(self, input: torch.Tensor, target: torch.Tensor) -> tuple:
        """
        Defines the way to process data.

        Parameters:
        - input_v (torch.Tensor): the input volume to process.
        - target_v (torch.Tensor): the target volume to process.

        Returns:
        - (torch.Tensor): the output tensor.
        """
        raise NotImplementedError()

    def __len__(self) -> int:
        return len(self._inputs)
