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
from .classification import ClassificationMetric


class Accuracy(ClassificationMetric):
    """
    Represents an accuracy metric.

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
        Instantiates a Accuracy.

        Parameters
        ----------
            params : Dict[str, int]
                parameters needed to adjust the metric behaviour
        """
        super(Accuracy, self).__init__(params)

        # Attributes
        if self._params["out_channels"] == 1:
            self._metric: torch.nn.Module = torchmetrics.Accuracy().to(self._DEVICE)

        else:
            self._metric: torch.nn.Module = torchmetrics.Accuracy(
                multiclass=True, num_classes=self._params["out_channels"], mdmc_reduce="samplewise"
            ).to(self._DEVICE)


class JaccardIndex(ClassificationMetric):
    """
    Represents a jaccard index metric.

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
        Instantiates a JaccardIndex.

        Parameters
        ----------
            params : Dict[str, int]
                parameters needed to adjust the metric behaviour
        """
        super(JaccardIndex, self).__init__(params)

        # Attributes
        if self._params["out_channels"] == 1:
            self._metric: torch.nn.Module = torchmetrics.JaccardIndex(
                num_classes=2
            ).to(self._DEVICE)

        else:
            self._metric: torch.nn.Module = torchmetrics.JaccardIndex(
                multiclass=True, num_classes=self._params["out_channels"], mdmc_reduce="samplewise"
            ).to(self._DEVICE)


class Precision(ClassificationMetric):
    """
    Represents a precision metric.

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
        Instantiates a Precision.

        Parameters
        ----------
            params : Dict[str, int]
                parameters needed to adjust the metric behaviour
        """
        super(Precision, self).__init__(params)

        # Attributes
        if self._params["out_channels"] == 1:
            self._metric: torch.nn.Module = torchmetrics.Precision().to(self._DEVICE)

        else:
            self._metric: torch.nn.Module = torchmetrics.Precision(
                multiclass=True, num_classes=self._params["out_channels"], mdmc_reduce="samplewise"
            ).to(self._DEVICE)


class Recall(ClassificationMetric):
    """
    Represents a recall metric.

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
        Instantiates a Recall.

        Parameters
        ----------
            params : Dict[str, int]
                parameters needed to adjust the metric behaviour
        """
        super(Recall, self).__init__(params)

        # Attributes
        if self._params["out_channels"] == 1:
            self._metric: torch.nn.Module = torchmetrics.Recall().to(self._DEVICE)

        else:
            self._metric: torch.nn.Module = torchmetrics.Recall(
                multiclass=True, num_classes=self._params["out_channels"], mdmc_reduce="samplewise"
            ).to(self._DEVICE)


class F1Score(ClassificationMetric):
    """
    Represents a f1score metric.

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
        Instantiates a F1Score.

        Parameters
        ----------
            params : Dict[str, int]
                parameters needed to adjust the metric behaviour
        """
        super(F1Score, self).__init__(params)

        # Attributes
        if self._params["out_channels"] == 1:
            self._metric: torch.nn.Module = torchmetrics.F1Score().to(self._DEVICE)

        else:
            self._metric: torch.nn.Module = torchmetrics.F1Score(
                multiclass=True, num_classes=self._params["out_channels"], mdmc_reduce="samplewise"
            ).to(self._DEVICE)
