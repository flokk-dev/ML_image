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

# IMPORT: project
from src.training.components.loss.loss import Loss


class RegressionLoss(Loss):
    def __init__(self):
        # Mother class
        super(RegressionLoss, self).__init__()

        # Attributes
        self._loss: typing.Any = None

    def __call__(
            self,
            prediction: torch.Tensor,
            target: torch.Tensor
    ) -> typing.Any:
        return self._loss(prediction, target)
