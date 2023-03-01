"""
Creator: HOCQUET Florian, Landza HOUDI
Date: 30/09/2022
Version: 1.0

Purpose:
"""

# IMPORT: deep learning
import torch
import torch.nn as nn

from monai.losses import FocalLoss, TverskyLoss


class UnifiedFocalLoss(nn.Module):
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self):
        super(UnifiedFocalLoss, self).__init__()

        self._focal_loss = FocalLoss().to(torch.device(self._DEVICE))
        self._tversky_loss = TverskyLoss().to(torch.device(self._DEVICE))

    def forward(self, prediction, ground_truth, weight=1):
        """
        Apply loss computation.

        Parameters:
        - prediction (torch.Tensor): the prediction.
        - ground_truth (torch.Tensor): the ground truth.
        """
        focal_loss_result = self._focal_loss(prediction, ground_truth)
        tversky_loss_result = self._tversky_loss(prediction, ground_truth)

        return weight*focal_loss_result + (1 - weight)*tversky_loss_result
