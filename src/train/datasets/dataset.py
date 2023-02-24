"""
Creator: HOCQUET Florian, Landza HOUDI
Date: 30/09/2022
Version: 1.0

Purpose: Articualtes the dataset's paths.
"""

# IMPORT: utils
import os
import random
import zstd

# IMPORT: data process
import numpy as np

# IMPORT: deep learning
import torch
from torch.utils.data import Dataset

import torchio as tio


class DataSet(Dataset):
    """
    The class representing a dataset.

    Attributes:
    - _dataset_info: the dataset information.
    - _transform: some transformations to apply to data.

    Methods:
    - _load_volume: Loads volume from path.
    - _my_collate: Defines the way to process data.
    """

    def __init__(self, params: dict, data_info: dict):
        """
        Initialize the DataSet class.

        Parameters:
        - data_info (dict): the ditionary needed to load the data.
        """
        self._params = params
        self._data_info = data_info

    def __getitem__(self, index: int) -> tuple:
        # LOADING
        shape = self._data_info["shapes"][index]

        input_volume = self._load_volume(self._data_info["inputs"][index], shape)
        target_volume = self._load_volume(self._data_info["targets"][index], shape)

        return self._my_collate(input_volume, target_volume)

    def _load_volume(self, path: str, shape: list) -> torch.Tensor:
        """
        Loads volume from path.

        Parameters:
        - path (str): the path of the volume to load.
        - shape (list): the shape of the volume.

        Returns:
        - (torch.Tensor): the output tensor.
        """
        file = np.load(path)

        decompressed_volume = zstd.decompress(file)
        flat_volume = np.frombuffer(decompressed_volume, dtype=np.float32).copy()
        volume = np.reshape(flat_volume, shape) 

        return torch.from_numpy(volume)

    def _my_collate(self, inputs: torch.Tensor, targets: torch.Tensor) -> tuple:
        """
        Defines the way to process data.

        Parameters:
        - input_v (torch.Tensor): the input volume to process.
        - target_v (torch.Tensor): the target volume to process.

        Returns:
        - (torch.Tensor): the output tensor.
        """
        raise NotImplementedError()

    @staticmethod
    def _get_nb_slices(input_volume: torch.Tensor, target_volume: torch.Tensor, n=0) -> tuple:
        """
        Get an equivalent number of segmented and unsegmented slices from targets.

        Parameters:
        - input_volume (torch.Tensor): one of the tensor to select slices from.
        - target_volume (torch.Tensor): one of the tensor to select slices from.

        Returns:
        - (tuple): the indexed input and target volumes.
        """
        if n > input_volume.shape[0]:
            n = input_volume.shape[0]

        idx = random.sample(range(input_volume.shape[0]), n)
        return input_volume[idx], target_volume[idx]

    def __len__(self) -> int:
        return len(self._data_info["inputs"])
