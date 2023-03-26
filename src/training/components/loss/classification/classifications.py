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
import monai

# IMPORT: project
from .classification import ClassificationLoss


class CELoss(ClassificationLoss):
    """
    Represents a cross entropy loss function.

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
        Instantiates a CELoss.

        Parameters
        ----------
            params : Dict[str, int]
                parameters needed to adjust the loss behaviour
        """
        # Mother class
        super(CELoss, self).__init__(params)

        # Attributes
        self._loss: torch.nn.Module = torch.nn.CrossEntropyLoss().to(self._DEVICE)
        self._behaviour: str = "minimization"


class BCELoss(ClassificationLoss):
    """
    Represents a binary cross entropy loss function.

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
        Instantiates a BCELoss.

        Parameters
        ----------
            params : Dict[str, int]
                parameters needed to adjust the loss behaviour
        """
        # Mother class
        super(BCELoss, self).__init__(params)

        # Attributes
        self._loss: torch.nn.Module = torch.nn.BCELoss().to(self._DEVICE)
        self._behaviour: str = "minimization"


class DiceLoss(ClassificationLoss):
    """
    Represents a dice loss function.

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
        Instantiates a DiceLoss.

        Parameters
        ----------
            params : Dict[str, int]
                parameters needed to adjust the loss behaviour
        """
        # Mother class
        super(DiceLoss, self).__init__(params)

        # Attributes
        self._loss: torch.nn.Module = monai.losses.DiceLoss().to(self._DEVICE)
        self._behaviour: str = "minimization"


class DiceCELoss(ClassificationLoss):
    """
    Represents a dice cross entropy loss function.

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
        Instantiates a DiceCELoss.

        Parameters
        ----------
            params : Dict[str, int]
                parameters needed to adjust the loss behaviour
        """
        # Mother class
        super(DiceCELoss, self).__init__(params)

        # Attributes
        self._loss: torch.nn.Module = monai.losses.DiceCELoss().to(self._DEVICE)
        self._behaviour: str = "minimization"


class DiceFocalLoss(ClassificationLoss):
    """
    Represents a dice focal loss function.

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
        Instantiates a DiceFocalLoss.

        Parameters
        ----------
            params : Dict[str, int]
                parameters needed to adjust the loss behaviour
        """
        # Mother class
        super(DiceFocalLoss, self).__init__(params)

        # Attributes
        self._loss: torch.nn.Module = monai.losses.DiceFocalLoss().to(self._DEVICE)
        self._behaviour: str = "minimization"


class FocalLoss(ClassificationLoss):
    """
    Represents a focal loss function.

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
        Instantiates a FocalLoss.

        Parameters
        ----------
            params : Dict[str, int]
                parameters needed to adjust the loss behaviour
        """
        # Mother class
        super(FocalLoss, self).__init__(params)

        # Attributes
        self._loss: torch.nn.Module = monai.losses.FocalLoss().to(self._DEVICE)
        self._behaviour: str = "minimization"


class FocalTverskyLoss(ClassificationLoss):
    """
    Represents a focal tversky loss function.

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
        Instantiates a FocalTverskyLoss.

        Parameters
        ----------
            params : Dict[str, int]
                parameters needed to adjust the loss behaviour
        """
        # Mother class
        super(FocalTverskyLoss, self).__init__(params)

        # Attributes
        self._loss: torch.nn.Module = monai.losses.TverskyLoss().to(self._DEVICE)
        self._behaviour: str = "minimization"
