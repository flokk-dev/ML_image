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
    def __init__(self):
        # Mother Class
        super(DataChopper2D, self).__init__()

    def _chopping(self, input_tensor, target_tensor=None):
        input_patches = torch.permute(input_tensor, (0, 2, 1, 3, 4)).squeeze(0)
        if target_tensor is None:
            return input_patches

        # TARGETS
        target_patches = torch.permute(target_tensor, (0, 2, 1, 3, 4)).squeeze(0)
        return input_patches, target_patches


class DataChopper25D(DataChopper):
    def __init__(self):
        # Mother Class
        super(DataChopper25D, self).__init__()

    def _chopping(self, input_tensor, target_tensor=None):
        input_patches = torch.squeeze(input_tensor).unfold(0, self._patch_height, 1)
        input_patches = torch.permute(input_patches, (0, 3, 1, 2))

        if target_tensor is None:
            return input_patches

        # TARGETS
        gap = self._patch_height // 2
        target_patches = torch.permute(target_tensor, (0, 2, 1, 3, 4)).squeeze(0)[gap:-gap]

        return input_patches, target_patches


class DataChopper3D(DataChopper):
    def __init__(self):
        # Mother Class
        super(DataChopper3D, self).__init__()

    def _chopping(self, input_tensor, target_tensor=None):
        raise NotImplementedError()
