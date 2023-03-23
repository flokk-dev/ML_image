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
    """
    Represents a mean absolute error metric.

    Attributes
    ----------
        _metric : torch.nn.Module
            loss function to apply.
    """
    def __init__(self):
        """ Instantiates a MAE. """
        super(MAE, self).__init__()

        # Attributes
        self._metric: torch.nn.Module = torch.nn.L1Loss().to(self._DEVICE)


class MSE(RegressionMetric):
    """
    Represents a mean squared error metric.

    Attributes
    ----------
        _metric : torch.nn.Module
            loss function to apply.
    """

    def __init__(self):
        """ Instantiates a MSE. """
        super(MSE, self).__init__()

        # Attributes
        self._metric: torch.nn.Module = torch.nn.MSELoss().to(self._DEVICE)


class RMSE(RegressionMetric):
    """
    Represents a root mean squared error metric.

    Attributes
    ----------
        _metric : torch.nn.Module
            loss function to apply.
    """

    def __init__(self):
        """ Instantiates a RMSE. """
        super(RMSE, self).__init__()

        # Attributes
        self._metric: torch.nn.Module = torch.nn.MSELoss().to(self._DEVICE)


class PSNR(RegressionMetric):
    """
    Represents a peak signal noise ratio metric.

    Attributes
    ----------
        _metric : torch.nn.Module
            loss function to apply.
    """

    def __init__(self):
        """ Instantiates a PSNR. """
        super(PSNR, self).__init__()

        # Attributes
        self._metric: torch.nn.Module = torchmetrics.PeakSignalNoiseRatio().to(self._DEVICE)


class SSIM(RegressionMetric):
    """
    Represents a structural similarity index measure metric.

    Attributes
    ----------
        _metric : torch.nn.Module
            loss function to apply.
    """

    def __init__(self):
        """ Instantiates a SSIM. """
        super(SSIM, self).__init__()

        # Attributes
        self._metric: torch.nn.Module = torchmetrics.StructuralSimilarityIndexMeasure().to(
            self._DEVICE
        )
