"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: deep learning
import torch

# IMPORT: project
from src.training.learner.loss.loss import Loss


class MAELoss(Loss):
    def __init__(self):
        super(MAELoss, self).__init__()

        # Attributes
        self._loss: torch.nn.Module = torch.nn.L1Loss().to(self._DEVICE)

    def __call__(
            self,
            prediction: torch.Tensor,
            target: torch.Tensor
    ) -> torch.Tensor:
        return self._loss(prediction, target)


class MSELoss(Loss):
    def __init__(self):
        super(MSELoss, self).__init__()

        # Attributes
        self._loss: torch.nn.Module = torch.nn.MSELoss().to(self._DEVICE)

    def __call__(
            self,
            prediction: torch.Tensor,
            target: torch.Tensor
    ) -> torch.Tensor:
        return self._loss(prediction, target)


class RMSELoss(Loss):
    def __init__(self):
        super(RMSELoss, self).__init__()

        # Attributes
        self._loss: torch.nn.Module = torch.nn.MSELoss().to(self._DEVICE)

    def __call__(
            self,
            prediction: torch.Tensor,
            target: torch.Tensor
    ) -> torch.Tensor:
        return torch.sqrt(self._loss(prediction, target))


class HuberLoss(Loss):
    def __init__(self):
        super(HuberLoss, self).__init__()

        # Attributes
        self._loss: torch.nn.Module = torch.nn.HuberLoss().to(self._DEVICE)

    def __call__(
            self,
            prediction: torch.Tensor,
            target: torch.Tensor
    ) -> torch.Tensor:
        return self._loss(prediction, target)
