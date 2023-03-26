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
from monai.networks import one_hot as one_hot_fn

# IMPORT: project
from src.training.components.metric.metric import Metric


class ClassificationMetric(Metric):
    """
    Represents a classification metric.

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
        Instantiates a ClassificationMetric.

        Parameters
        ----------
            params : Dict[str, int]
                parameters needed to adjust the metric behaviour
        """
        # Mother class
        super(ClassificationMetric, self).__init__(params)

    def __call__(self, prediction_batch: torch.Tensor, target_batch: torch.Tensor = None) \
            -> torch.Tensor:
        """
        Parameters
        ----------
            prediction_batch : torch.Tensor
                batch of predicted tensors
            target_batch : torch.Tensor
                batch of target tensors

        Returns
        ----------
            torch.Tensor
                value of the loss applied to the prediction and the target
        """
        if target_batch is None:
            return self._metric(prediction_batch)

        if target_batch.shape[1] < prediction_batch.shape[1]:
            target_batch = one_hot_fn(labels=target_batch, num_classes=prediction_batch.shape[1])

        target_batch = target_batch.type(torch.int32)
        return self._metric(prediction_batch, target_batch)
