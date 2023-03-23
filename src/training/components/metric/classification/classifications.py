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
from .classification import ClassificationMetric


class Accuracy(ClassificationMetric):
    """
    Represents an accuracy metric.

    Attributes
    ----------
        _metric : torch.nn.Module
            loss function to apply.
    """

    def __init__(self):
        """ Instantiates a Accuracy. """
        super(Accuracy, self).__init__()

        # Attributes
        self._metric: torch.nn.Module = torchmetrics.Accuracy().to(self._DEVICE)


class JaccardIndex(ClassificationMetric):
    """
    Represents a jaccard index metric.

    Attributes
    ----------
        _metric : torch.nn.Module
            loss function to apply.
    """

    def __init__(self):
        """ Instantiates a JaccardIndex. """
        super(JaccardIndex, self).__init__()

        # Attributes
        self._metric: torch.nn.Module = torchmetrics.JaccardIndex(num_classes=2).to(self._DEVICE)


class Precision(ClassificationMetric):
    """
    Represents a precision metric.

    Attributes
    ----------
        _metric : torch.nn.Module
            loss function to apply.
    """

    def __init__(self):
        """ Instantiates a Precision. """
        super(Precision, self).__init__()

        # Attributes
        self._metric: torch.nn.Module = torchmetrics.Precision().to(self._DEVICE)


class Recall(ClassificationMetric):
    """
    Represents a recall metric.

    Attributes
    ----------
        _metric : torch.nn.Module
            loss function to apply.
    """

    def __init__(self):
        """ Instantiates a Recall. """
        super(Recall, self).__init__()

        # Attributes
        self._metric: torch.nn.Module = torchmetrics.Recall().to(self._DEVICE)


class F1Score(ClassificationMetric):
    """
    Represents a f1score metric.

    Attributes
    ----------
        _metric : torch.nn.Module
            loss function to apply.
    """

    def __init__(self):
        """ Instantiates a F1Score. """
        super(F1Score, self).__init__()

        # Attributes
        self._metric: torch.nn.Module = torchmetrics.F1Score().to(self._DEVICE)
