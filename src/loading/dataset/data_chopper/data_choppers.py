"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import typing

# IMPORT: data processing
import torch

# IMPORT: projet
from .data_chopper import DataChopper


class DataChopper2D(DataChopper):
    def __init__(self):
        """
        pass.
        """
        # Mother Class
        super(DataChopper2D, self).__init__()

    def _chop(
            self,
            input_t: torch.Tensor,
            target_t: torch.Tensor = None
    ) -> [torch.Tensor, typing.Tuple[torch.Tensor, torch.Tensor]]:
        """
        pass.
        """
        input_p: torch.Tensor = input_t.squeeze(0).permute((1, 0, 2, 3))
        if target_t is None:
            return input_p

        # TARGETS
        target_p: torch.Tensor = target_t.squeeze(0).permute((1, 0, 2, 3))
        return input_p, target_p


class DataChopper25D(DataChopper):
    def __init__(self):
        """
        pass.
        """
        # Mother Class
        super(DataChopper25D, self).__init__()

        # Attributes
        self._patch_height = 5

    def _chop(
            self,
            input_t: torch.Tensor,
            target_t: torch.Tensor = None
    ) -> [torch.Tensor, typing.Tuple[torch.Tensor, torch.Tensor]]:
        """
        pass.
        """
        input_t: torch.Tensor = input_t.squeeze(0).squeeze(0)
        input_p: torch.Tensor = input_t.unfold(0, self._patch_height, 1).permute((0, 3, 1, 2))

        if target_t is None:
            return input_p

        # TARGETS
        gap: int = self._patch_height // 2
        target_p: torch.Tensor = target_t.squeeze(0).permute((1, 0, 2, 3))[gap:-gap]

        return input_p, target_p


class DataChopper3D(DataChopper):
    def __init__(self):
        """
        pass.
        """
        # Mother Class
        super(DataChopper3D, self).__init__()

    def _chop(
            self,
            input_t: torch.Tensor,
            target_t: torch.Tensor = None
    ) -> typing.Any:
        """
        pass.
        """
        raise NotImplementedError()
