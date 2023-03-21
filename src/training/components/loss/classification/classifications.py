"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: deep learning
import torch
import monai

# IMPORT: project
from .classification import ClassificationLoss


class CELoss(ClassificationLoss):
    def __init__(self):
        # Mother class
        super(CELoss, self).__init__()

        # Attributes
        self._loss: torch.nn.Module = torch.nn.CrossEntropyLoss().to(self._DEVICE)


class BCELoss(ClassificationLoss):
    def __init__(self):
        # Mother class
        super(BCELoss, self).__init__()

        # Attributes
        self._loss: torch.nn.Module = torch.nn.BCELoss().to(self._DEVICE)


class DiceLoss(ClassificationLoss):
    def __init__(self):
        # Mother class
        super(DiceLoss, self).__init__()

        # Attributes
        self._loss: torch.nn.Module = monai.losses.DiceLoss().to(self._DEVICE)


class DiceCELoss(ClassificationLoss):
    def __init__(self):
        # Mother class
        super(DiceCELoss, self).__init__()

        # Attributes
        self._loss: torch.nn.Module = monai.losses.DiceCELoss().to(self._DEVICE)


class DiceFocalLoss(ClassificationLoss):
    def __init__(self):
        # Mother class
        super(DiceFocalLoss, self).__init__()

        # Attributes
        self._loss: torch.nn.Module = monai.losses.DiceFocalLoss().to(self._DEVICE)


class FocalLoss(ClassificationLoss):
    def __init__(self):
        # Mother class
        super(FocalLoss, self).__init__()

        # Attributes
        self._loss: torch.nn.Module = monai.losses.FocalLoss().to(self._DEVICE)


class FocalTverskyLoss(ClassificationLoss):
    def __init__(self):
        # Mother class
        super(FocalTverskyLoss, self).__init__()

        # Attributes
        self._loss: torch.nn.Module = monai.losses.TverskyLoss().to(self._DEVICE)
