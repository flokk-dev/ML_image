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


class DataChopper:
    def __init__(self):
        """
        pass.
        """
        pass

    def _chop(
            self,
            input_t: torch.Tensor,
            target_t: torch.Tensor = None
    ) -> typing.Any:
        """
        pass.
        """
        raise NotImplementedError()

    def __call__(
            self,
            input_t: torch.Tensor,
            target_t: torch.Tensor = None
    ) -> typing.Any:
        """
        pass.
        """
        return self._chop(input_t, target_t)
