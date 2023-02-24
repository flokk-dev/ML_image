"""
Creator: HOCQUET Florian, Landza HOUDI
Date: 30/09/2022
Version: 1.0

Purpose: Represents the data in 2D.
"""

# IMPORT: deep learning
import torch

# IMPORT: projet
from .dataset import DataSet


class DataSet2D(DataSet):
    """
    The class representing a 2D dataset.

    Attributes:
    - _transform: some transformations to apply to data.

    Methods:
    - _my_collate: Defines the way to process data.
    """

    def __init__(self, params: dict, data_info: dict, info: dict):
        """
        Initialize the DataSet2D class.

        Parameters:
        - data_info (dict): the dataset information.
        - _inputs (list): the input datas' path.
        - _targets (list): the target datas' path.
        """
        # Mother Class
        super(DataSet2D, self).__init__(params, data_info, info)

        # Attributes
        self._params = params

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
        input_patches = torch.squeeze(input)
        input_patches = input_patches.unfold(0, 1, 1)
        input_patches = torch.movedim(input_patches, 3, 1)

        # TARGETS
        target_patches = torch.movedim(target, 1, 0)

        return input_patches, target_patches
