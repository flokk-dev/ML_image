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
from src.training.components.metric.metric import Metric


class Accuracy(Metric):
    def __init__(self):
        super(Accuracy, self).__init__()

        # Attributes
        self._metric = torchmetrics.Accuracy().to(self._DEVICE)

    def __call__(self, prediction, target):
        return self._loss(prediction, target)


class JaccardIndex(Metric):
    def __init__(self):
        super(JaccardIndex, self).__init__()

        # Attributes
        self._loss = torchmetrics.JaccardIndex(num_classes=2).to(self._DEVICE)

    def __call__(self, prediction, target):
        return self._loss(prediction, target)


class Precision(Metric):
    def __init__(self):
        super(Precision, self).__init__()

        # Attributes
        self._loss = torchmetrics.Precision().to(self._DEVICE)

    def __call__(self, prediction, target):
        return torch.sqrt(self._loss(prediction, target))


class Recall(Metric):
    def __init__(self):
        super(Recall, self).__init__()

        # Attributes
        self._loss = torchmetrics.Recall().to(self._DEVICE)

    def __call__(self, prediction, target):
        return self._loss(prediction, target)


class F1Score(Metric):
    def __init__(self):
        super(F1Score, self).__init__()

        # Attributes
        self._loss = torchmetrics.F1Score().to(self._DEVICE)

    def __call__(self, prediction, target):
        return self._loss(prediction, target)
