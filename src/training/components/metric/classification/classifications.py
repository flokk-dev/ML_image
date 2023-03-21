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
    def __init__(self):
        super(Accuracy, self).__init__()

        # Attributes
        self._metric: torch.nn.Module = torchmetrics.Accuracy().to(self._DEVICE)


class JaccardIndex(ClassificationMetric):
    def __init__(self):
        super(JaccardIndex, self).__init__()

        # Attributes
        self._metric: torch.nn.Module = torchmetrics.JaccardIndex(num_classes=2).to(self._DEVICE)


class Precision(ClassificationMetric):
    def __init__(self):
        super(Precision, self).__init__()

        # Attributes
        self._metric: torch.nn.Module = torchmetrics.Precision().to(self._DEVICE)


class Recall(ClassificationMetric):
    def __init__(self):
        super(Recall, self).__init__()

        # Attributes
        self._metric: torch.nn.Module = torchmetrics.Recall().to(self._DEVICE)


class F1Score(ClassificationMetric):
    def __init__(self):
        super(F1Score, self).__init__()

        # Attributes
        self._metric: torch.nn.Module = torchmetrics.F1Score().to(self._DEVICE)
