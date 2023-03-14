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

    def _chop(self, input_tensor, target_tensor=None):
        input_patches = input_tensor.squeeze(0).permute((1, 0, 2, 3))
        if target_tensor is None:
            return input_patches

        # TARGETS
        target_patches = target_tensor.squeeze(0).permute((1, 0, 2, 3))
        return input_patches, target_patches


class DataChopper25D(DataChopper):
    def __init__(self):
        # Mother Class
        super(DataChopper25D, self).__init__()

        # Attributes
        self._patch_height = 5

    def _chop(self, input_tensor, target_tensor=None):
        input_tensor = input_tensor.squeeze(0).squeeze(0)
        input_patches = input_tensor.unfold(0, self._patch_height, 1).permute((0, 3, 1, 2))

        if target_tensor is None:
            return input_patches

        # TARGETS
        gap = self._patch_height // 2
        target_patches = target_tensor.squeeze(0).permute((1, 0, 2, 3))[gap:-gap]

        return input_patches, target_patches


class DataChopper3D(DataChopper):
    def __init__(self):
        # Mother Class
        super(DataChopper3D, self).__init__()

    def _chop(self, input_tensor, target_tensor=None):
        raise NotImplementedError()
