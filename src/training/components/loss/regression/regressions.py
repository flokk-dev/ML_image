"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *

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
        _behaviour: str
            loss' behaviour
        _params : Dict[str, int]
            parameters needed to adjust the loss behaviour
    """

    def __init__(self, params: Dict[str, int]):
        """
        Instantiates a MAELoss.

        Parameters
        ----------
            params : Dict[str, int]
                parameters needed to adjust the loss behaviour
        """
        # Mother class
        super(MAELoss, self).__init__(params)

        # Attributes
        self._loss: torch.nn.Module = torch.nn.L1Loss().to(self._DEVICE)
        self._behaviour: str = "minimization"


class MSELoss(RegressionLoss):
    """
    Represents a mean squared error loss function.

    Attributes
    ----------
        _loss : torch.nn.Module
            loss function to apply.
        _behaviour: str
            loss' behaviour
        _params : Dict[str, int]
            parameters needed to adjust the loss behaviour
    """

    def __init__(self, params: Dict[str, int]):
        """
        Instantiates a MSELoss.

        Parameters
        ----------
            params : Dict[str, int]
                parameters needed to adjust the loss behaviour
        """
        # Mother class
        super(MSELoss, self).__init__(params)

        # Attributes
        self._loss: torch.nn.Module = torch.nn.MSELoss().to(self._DEVICE)
        self._behaviour: str = "minimization"


class RMSELoss(RegressionLoss):
    """
    Represents a root mean squared error loss function.

    Attributes
    ----------
        _loss : torch.nn.Module
            loss function to apply.
        _behaviour: str
            loss' behaviour
        _params : Dict[str, int]
            parameters needed to adjust the loss behaviour
    """

    def __init__(self, params: Dict[str, int]):
        """
        Instantiates a RMSELoss.

        Parameters
        ----------
            params : Dict[str, int]
                parameters needed to adjust the loss behaviour
        """
        # Mother class
        super(RMSELoss, self).__init__(params)

        # Attributes
        self._loss: torch.nn.Module = torch.nn.MSELoss().to(self._DEVICE)
        self._behaviour: str = "minimization"


class HuberLoss(RegressionLoss):
    """
    Represents a huber loss function.

    Attributes
    ----------
        _loss : torch.nn.Module
            loss function to apply.
        _behaviour: str
            loss' behaviour
        _params : Dict[str, int]
            parameters needed to adjust the loss behaviour
    """

    def __init__(self, params: Dict[str, int]):
        """
        Instantiates a HuberLoss.

        Parameters
        ----------
            params : Dict[str, int]
                parameters needed to adjust the loss behaviour
        """
        # Mother class
        super(HuberLoss, self).__init__(params)

        # Attributes
        self._loss: torch.nn.Module = torch.nn.HuberLoss().to(self._DEVICE)
        self._behaviour: str = "minimization"
