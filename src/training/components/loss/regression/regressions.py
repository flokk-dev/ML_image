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
    """
    Represents a mean absolute error loss function.

    Attributes
    ----------
        _loss : torch.nn.Module
            loss function to apply.
    """

    def __init__(self):
        """ Instantiates a MAELoss. """
        super(MAELoss, self).__init__()

        # Attributes
        self._loss: torch.nn.Module = torch.nn.L1Loss().to(self._DEVICE)


class MSELoss(RegressionLoss):
    """
    Represents a mean squared error loss function.

    Attributes
    ----------
        _loss : torch.nn.Module
            loss function to apply.
    """

    def __init__(self):
        """ Instantiates a MSELoss. """
        super(MSELoss, self).__init__()

        # Attributes
        self._loss: torch.nn.Module = torch.nn.MSELoss().to(self._DEVICE)


class RMSELoss(RegressionLoss):
    """
    Represents a root mean squared error loss function.

    Attributes
    ----------
        _loss : torch.nn.Module
            loss function to apply.
    """

    def __init__(self):
        """ Instantiates a RMSELoss. """
        super(RMSELoss, self).__init__()

        # Attributes
        self._loss: torch.nn.Module = torch.nn.MSELoss().to(self._DEVICE)


class HuberLoss(RegressionLoss):
    """
    Represents a huber loss function.

    Attributes
    ----------
        _loss : torch.nn.Module
            loss function to apply.
    """

    def __init__(self):
        """ Instantiates a HuberLoss. """
        super(HuberLoss, self).__init__()

        # Attributes
        self._loss: torch.nn.Module = torch.nn.HuberLoss().to(self._DEVICE)
