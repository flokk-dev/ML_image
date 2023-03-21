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
from monai.networks import one_hot as one_hot_fn

# IMPORT: project
from src.training.components.metric.metric import Metric


class ClassificationMetric(Metric):
    def __init__(self):
        # Mother class
        super(ClassificationMetric, self).__init__()

        # Attributes
        self._metric: typing.Any = None

    def __call__(
            self,
            prediction: torch.Tensor,
            target: torch.Tensor
    ) -> typing.Any:
        target = one_hot_fn(labels=target, num_classes=prediction.shape[1], dtype=torch.int32)
        return self._metric(prediction, target)