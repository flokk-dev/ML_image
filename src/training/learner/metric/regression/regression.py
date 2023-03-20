"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: deep learning
import torch
import torchmetrics

# IMPORT: project
from src.training.learner.metric.metric import Metric


class MAE(Metric):
    def __init__(self):
        super(MAE, self).__init__()

        # Attributes
        self._loss: torch.nn.Module = torch.nn.L1Loss().to(self._DEVICE)

    def __call__(
            self,
            prediction: torch.Tensor,
            target: torch.Tensor
    ) -> torch.Tensor:
        return self._loss(prediction, target)


class MSE(Metric):
    def __init__(self):
        super(MSE, self).__init__()

        # Attributes
        self._loss: torch.nn.Module = torch.nn.MSELoss().to(self._DEVICE)

    def __call__(
            self,
            prediction: torch.Tensor,
            target: torch.Tensor
    ) -> torch.Tensor:
        return self._loss(prediction, target)


class RMSE(Metric):
    def __init__(self):
        super(RMSE, self).__init__()

        # Attributes
        self._loss: torch.nn.Module = torch.nn.MSELoss().to(self._DEVICE)

    def __call__(
            self,
            prediction: torch.Tensor,
            target: torch.Tensor
    ) -> torch.Tensor:
        return torch.sqrt(self._loss(prediction, target))


class PSNR(Metric):
    def __init__(self):
        super(PSNR, self).__init__()

        # Attributes
        self._loss: torch.nn.Module = torchmetrics.PeakSignalNoiseRatio().to(self._DEVICE)

    def __call__(
            self,
            prediction: torch.Tensor,
            target: torch.Tensor
    ) -> torch.Tensor:
        return torch.sqrt(self._loss(prediction, target))


class SSIM(Metric):
    def __init__(self):
        super(SSIM, self).__init__()

        # Attributes
        self._loss: torch.nn.Module = torchmetrics.StructuralSimilarityIndexMeasure().to(self._DEVICE)

    def __call__(
            self,
            prediction: torch.Tensor,
            target: torch.Tensor
    ) -> torch.Tensor:
        return torch.sqrt(self._loss(prediction, target))
