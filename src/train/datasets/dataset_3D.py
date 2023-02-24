"""
Creator: HOCQUET Florian, Landza HOUDI
Date: 30/09/2022
Version: 1.0

Purpose: Represents the data in 3D.
"""

# IMPORT: deep learning
import torch

import torchio as tio
from torchio.data import LabelSampler

# IMPORT: projet
import utils
from .dataset import DataSet


class DataSet3D(DataSet):
    """
    The class representing a 3D dataset.

    Attributes:
    - _transform: some transformations to apply to data.

    Methods:
    - _my_collate: Defines the way to process data.
    """

    def __init__(self, params: dict, data_info: dict):
        """
        Initialize the DataSet3D class.

        Parameters:
        - data_info (dict): the dataset information.
        - _inputs (list): the input datas' path.
        - _targets (list): the target datas' path.
        """
        # Mother Class
        super(DataSet3D, self).__init__(params, data_info)

        # Attributes
        self._params = params
        self._transform = lambda x: tio.CropOrPad((16 * ((x.shape[1] // 16) + 1), 512, 512))(x)

        self._patch_shape = (64, 64, 64)
        self._probs = {0: 0.5, 1: 0.5}

    def _load_volume(self, path: str, shape: list = 0) -> torch.Tensor:
        """
        Loads volume from path.

        Parameters:
        - path (str): the path of the volume to load.
        - shape (str): the shape of the volume.
        """
        volume = super()._load_volume(path, shape).type(torch.float32)
        return self._transform(volume).data.type(torch.float16)

    def _my_collate(self, input_volume: torch.Tensor, target_volume: torch.Tensor) -> tuple:
        """
        Defines the way to process data.

        Parameters:
        - input_v (torch.Tensor): the input volume to process.
        - target_v (torch.Tensor): the target volume to process.

        Returns:
        - (torch.Tensor): the output tensors.
        """
        # INPUTS AND TARGETS
        input_subject = tio.ScalarImage(tensor=input_volume)
        target_subject = tio.LabelMap(tensor=target_volume)

        subject = tio.Subject({"input": input_subject, "target": target_subject})
        sampler = LabelSampler(patch_size=self._patch_shape, label_probabilities=self._probs)

        input_patches, target_patches = list(zip(
            *[(utils.add_dim(patche["input"].data), utils.add_dim(patche["target"].data))
                for patche in sampler(subject, num_patches=250)]
        ))

        return torch.cat(input_patches, dim=0), torch.cat(target_patches, dim=0)
