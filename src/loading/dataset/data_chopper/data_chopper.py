"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: dataset processing
import torch


class DataChopper:
    def __init__(self, params: dict):
        # Attributes
        self._params = params

    def _chopping(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> tuple:
        raise NotImplementedError()

    def __call__(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> tuple:
        return self._chopping(input_tensor, target_tensor)
