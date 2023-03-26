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
import torchmetrics

# IMPORT: project
from .regression import RegressionMetric


class MAE(RegressionMetric):
    """
    Represents a mean absolute error metric.

    Attributes
    ----------
        _metric : torch.nn.Module
            metric to apply.
        _behaviour: str
            metric's behaviour
        _params : Dict[str, int]
            parameters needed to adjust the metric behaviour
    """

    def __init__(self, params: Dict[str, int]):
        """
        Instantiates a MAE.

        Parameters
        ----------
            params : Dict[str, int]
                parameters needed to adjust the metric behaviour
        """
        super(MAE, self).__init__(params)

        # Attributes
        self._metric: torch.nn.Module = torch.nn.L1Loss().to(self._DEVICE)


class MSE(RegressionMetric):
    """
    Represents a mean squared error metric.

    Attributes
    ----------
        _metric : torch.nn.Module
            metric to apply.
        _behaviour: str
            metric's behaviour
        _params : Dict[str, int]
            parameters needed to adjust the metric behaviour
    """

    def __init__(self, params: Dict[str, int]):
        """
        Instantiates a MSE.

        Parameters
        ----------
            params : Dict[str, int]
                parameters needed to adjust the metric behaviour
        """
        super(MSE, self).__init__(params)

        # Attributes
        self._metric: torch.nn.Module = torch.nn.MSELoss().to(self._DEVICE)


class RMSE(RegressionMetric):
    """
    Represents a root mean squared error metric.

    Attributes
    ----------
        _metric : torch.nn.Module
            metric to apply.
        _behaviour: str
            metric's behaviour
        _params : Dict[str, int]
            parameters needed to adjust the metric behaviour
    """

    def __init__(self, params: Dict[str, int]):
        """
        Instantiates a RMSE.

        Parameters
        ----------
            params : Dict[str, int]
                parameters needed to adjust the metric behaviour
        """
        super(RMSE, self).__init__(params)

        # Attributes
        self._metric: torch.nn.Module = torch.nn.MSELoss().to(self._DEVICE)


class PSNR(RegressionMetric):
    """
    Represents a peak signal noise ratio metric.

    Attributes
    ----------
        _metric : torch.nn.Module
            metric to apply.
        _behaviour: str
            metric's behaviour
        _params : Dict[str, int]
            parameters needed to adjust the metric behaviour
    """

    def __init__(self, params: Dict[str, int]):
        """
        Instantiates a PSNR.

        Parameters
        ----------
            params : Dict[str, int]
                parameters needed to adjust the metric behaviour
        """
        super(PSNR, self).__init__(params)

        # Attributes
        self._metric: torch.nn.Module = torchmetrics.PeakSignalNoiseRatio().to(self._DEVICE)


class SSIM(RegressionMetric):
    """
    Represents a structural similarity index measure metric.

    Attributes
    ----------
        _metric : torch.nn.Module
            metric to apply.
        _behaviour: str
            metric's behaviour
        _params : Dict[str, int]
            parameters needed to adjust the metric behaviour
    """

    def __init__(self, params: Dict[str, int]):
        """
        Instantiates a SSIM.

        Parameters
        ----------
            params : Dict[str, int]
                parameters needed to adjust the metric behaviour
        """
        super(SSIM, self).__init__(params)

        # Attributes
        self._metric: torch.nn.Module = torchmetrics.StructuralSimilarityIndexMeasure().to(
            self._DEVICE
        )
