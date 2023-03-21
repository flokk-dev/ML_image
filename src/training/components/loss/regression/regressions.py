"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: deep learning
import torch

# IMPORT: project
from .regression import RegressionLoss


class MAELoss(RegressionLoss):
    def __init__(self):
        super(MAELoss, self).__init__()

        # Attributes
        self._loss: torch.nn.Module = torch.nn.L1Loss().to(self._DEVICE)


class MSELoss(RegressionLoss):
    def __init__(self):
        super(MSELoss, self).__init__()

        # Attributes
        self._loss: torch.nn.Module = torch.nn.MSELoss().to(self._DEVICE)


class RMSELoss(RegressionLoss):
    def __init__(self):
        super(RMSELoss, self).__init__()

        # Attributes
        self._loss: torch.nn.Module = torch.nn.MSELoss().to(self._DEVICE)


class HuberLoss(RegressionLoss):
    def __init__(self):
        super(HuberLoss, self).__init__()

        # Attributes
        self._loss: torch.nn.Module = torch.nn.HuberLoss().to(self._DEVICE)
