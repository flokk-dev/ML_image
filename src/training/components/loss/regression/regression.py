"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: deep learning
import torch

# IMPORT: project
from src.training.components.loss.loss import Loss


class RegressionLoss(Loss):
    """
    Represents a regression loss function.

    Attributes
    ----------
        _loss : torch.nn.Module
            loss function to apply.
    """

    def __init__(self):
        """ Instantiates a RegressionLoss. """
        # Mother class
        super(RegressionLoss, self).__init__()

        # Attributes
        self._loss: torch.nn.Module = None

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
            return self._loss(prediction_batch)

        return self._loss(prediction_batch, target_batch)
