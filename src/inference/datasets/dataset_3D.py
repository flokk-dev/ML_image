"""
Creator: HOCQUET Florian, Landza HOUDI
Date: 30/09/2022
Version: 1.0

Purpose: Represents the data in 3D.
"""

# IMPORT: deep learning
import torch
import torchio as tio

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

    def __init__(self, params: dict, data_info: dict, info: dict):
        """
        Initialize the DataSet3D class.

        Parameters:
        - data_info (dict): the dataset information.
        - _inputs (list): the input datas' path.
        - _targets (list): the target datas' path.
        """
        # Mother Class
        super(DataSet3D, self).__init__(params, data_info, info)

        # Attributes
        self._params = params
        self._transform = lambda x: tio.CropOrPad((64 * ((x.shape[1] // 64) + 1), 512, 512))(x)

        self._patch_shape = (64, 64, 64)
        self._probs = {0: 0.5, 1: 0.5}

    def _my_collate(self, input: torch.Tensor, target: torch.Tensor) -> tuple:
        """
        Defines the way to process data.

        Parameters:
        - input_v (torch.Tensor): the input volume to process.
        - target_v (torch.Tensor): the target volume to process.

        Returns:
        - (torch.Tensor): the output tensors.
        """
        # INPUTS
        input = utils.pad_tensor(input)

        input_patches = list()
        z, x, y = self._patch_shape

        chunks = torch.split(input, z, dim=1)
        for chunk_i in chunks:
            chunks_x = torch.split(chunk_i, x, dim=2)
            for chunk_x_i in chunks_x:
                chunk_y_i = torch.split(chunk_x_i, y, dim=3)
                for final_chunk in chunk_y_i:
                    input_patches.append(utils.add_dim(final_chunk))

        input_patches = torch.cat(input_patches, dim=0)
        input_patches = (input_patches, self._patch_shape, input.shape, target.shape[1])

        # TARGETS
        target_patches = torch.movedim(target, 1, 0)

        return input_patches, target_patches
