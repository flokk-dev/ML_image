"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: data processing
import torch

# IMPORT: projet
from .data_chopper import DataChopper


class DataChopper2D(DataChopper):
    def __init__(self, params: dict):
        # Mother Class
        super(DataChopper2D, self).__init__(params)

    def _chopping(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> tuple:
        # INPUTS
        input_patches = torch.squeeze(input_tensor).unfold(0, 1, 1)
        input_patches = torch.movedim(input_patches, 3, 1)

        # TARGETS
        target_patches = torch.squeeze(target_tensor).unfold(0, 1, 1)
        target_patches = torch.movedim(target_patches, 3, 1)

        return input_patches, target_patches


class DataChopper25D(DataChopper):
    def __init__(self, params: dict):
        # Mother Class
        super(DataChopper25D, self).__init__(params)

    def _chopping(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> tuple:
        # INPUTS
        input_patches = torch.squeeze(input_tensor)
        input_patches = input_patches.unfold(0, self._params["patch_height"], 1)
        input_patches = torch.movedim(input_patches, 3, 1)

        # TARGETS
        gap = self._params["patch_height"] // 2
        target_patches = torch.movedim(target_tensor, 1, 0)[gap:-gap]

        return input_patches, target_patches


class DataChopper3D(DataChopper):
    def __init__(self, params: dict):
        # Mother Class
        super(DataChopper3D, self).__init__(params)

    def _chopping(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> tuple:
        raise NotImplementedError()
