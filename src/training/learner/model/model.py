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

    def __init__(self, data_info, weights_path=None):
        # Attributes
        self._data_info = data_info
        self._weights_path = weights_path
