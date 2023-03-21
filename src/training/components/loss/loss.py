"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import typing

# IMPORT: deep learning
import torch


class Loss:
    _DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def __init__(self):
        # Attributes
        self._loss: typing.Any = None

    def __call__(
            self,
            prediction: torch.Tensor,
            target: torch.Tensor
    ) -> typing.Any:
        raise NotImplementedError()
