"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: deep learning
import torch


class Model:
    _DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def __init__(self, input_shape, target_shape, weights_path=None):
        # Attributes
        self._params = self._parse_shape(input_shape, target_shape)
        self._weights_path = weights_path

    @staticmethod
    def _parse_shape(input_shape, target_shape):
        return {
            "spatial_dims": len(input_shape) - 2,
            "img_size": tuple(input_shape[2:]),
            "in_channels": input_shape[1],
            "out_channels": target_shape[1]
        }
