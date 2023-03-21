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
from .regression import RegressionMetric


class MAE(RegressionMetric):
    def __init__(self):
        super(MAE, self).__init__()

        # Attributes
        self._metric: torch.nn.Module = torch.nn.L1Loss().to(self._DEVICE)


class MSE(RegressionMetric):
    def __init__(self):
        super(MSE, self).__init__()

        # Attributes
        self._metric: torch.nn.Module = torch.nn.MSELoss().to(self._DEVICE)


class RMSE(RegressionMetric):
    def __init__(self):
        super(RMSE, self).__init__()

        # Attributes
        self._metric: torch.nn.Module = torch.nn.MSELoss().to(self._DEVICE)


class PSNR(RegressionMetric):
    def __init__(self):
        super(PSNR, self).__init__()

        # Attributes
        self._metric: torch.nn.Module = torchmetrics.PeakSignalNoiseRatio().to(self._DEVICE)


class SSIM(RegressionMetric):
    def __init__(self):
        super(SSIM, self).__init__()

        # Attributes
        self._metric: torch.nn.Module = torchmetrics.StructuralSimilarityIndexMeasure().to(self._DEVICE)
