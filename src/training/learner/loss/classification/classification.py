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
import monai

# IMPORT: project
from src.training.learner.loss.loss import Loss, CompositeLoss


class CELoss(Loss):
    def __init__(self):
        super(CELoss, self).__init__()

        # Attributes
        self._loss: torch.nn.Module = torch.nn.CrossEntropyLoss().to(self._DEVICE)

    def __call__(
            self,
            prediction: torch.Tensor,
            target: torch.Tensor
    ) -> torch.Tensor:
        return self._loss(prediction, target)


class BCELoss(Loss):
    def __init__(self):
        super(BCELoss, self).__init__()

        # Attributes
        self._loss: torch.nn.Module = torch.nn.BCELoss().to(self._DEVICE)

    def __call__(
            self,
            prediction: torch.Tensor,
            target: torch.Tensor
    ) -> torch.Tensor:
        return self._loss(prediction, target)


class DiceLoss(Loss):
    def __init__(self):
        super(DiceLoss, self).__init__()

        # Attributes
        self._loss: torch.nn.Module = monai.losses.DiceLoss().to(self._DEVICE)

    def __call__(
            self,
            prediction: torch.Tensor,
            target: torch.Tensor
    ) -> torch.Tensor:
        return self._loss(prediction, target)


class DiceCELoss(Loss):
    def __init__(self):
        super(DiceCELoss, self).__init__()

        # Attributes
        self._loss: torch.nn.Module = monai.losses.DiceCELoss().to(self._DEVICE)

    def __call__(
            self,
            prediction: torch.Tensor,
            target: torch.Tensor
    ) -> torch.Tensor:
        return self._loss(prediction, target)


class DiceFocalLoss(Loss):
    def __init__(self):
        super(DiceFocalLoss, self).__init__()

        # Attributes
        self._loss: torch.nn.Module = monai.losses.DiceFocalLoss().to(self._DEVICE)

    def __call__(
            self,
            prediction: torch.Tensor,
            target: torch.Tensor
    ) -> torch.Tensor:
        return self._loss(prediction, target)


class FocalLoss(Loss):
    def __init__(self):
        super(FocalLoss, self).__init__()

        # Attributes
        self._loss: torch.nn.Module = monai.losses.FocalLoss().to(self._DEVICE)

    def __call__(
            self,
            prediction: torch.Tensor,
            target: torch.Tensor
    ) -> torch.Tensor:
        return self._loss(prediction, target)


class FocalTverskyLoss(Loss):
    def __init__(self):
        super(FocalTverskyLoss, self).__init__()

        # Attributes
        self._loss: torch.nn.Module = monai.losses.TverskyLoss().to(self._DEVICE)

    def __call__(
            self,
            prediction: torch.Tensor,
            target: torch.Tensor
    ) -> torch.Tensor:
        return self._loss(prediction, target)


class UnifiedFocalLoss(CompositeLoss):
    def __init__(self, weights):
        super(UnifiedFocalLoss, self).__init__()

        # Attributes
        self._losses: typing.List[typing.Dict[str, typing.Any]] = [
            {"loss": monai.losses.FocalLoss().to(self._DEVICE), "weight": weights[0]},
            {"loss": monai.losses.TverskyLoss().to(self._DEVICE), "weight": weights[0]},
        ]
        self._verify_weights(weights)

    def __call__(
            self,
            prediction: torch.Tensor,
            target: torch.Tensor
    ) -> torch.Tensor:
        loss_value: float = 0.0
        for loss, weight in self._losses:
            loss_value += loss(prediction, target) * weight

        return loss_value
