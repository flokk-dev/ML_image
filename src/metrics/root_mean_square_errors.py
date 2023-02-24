"""
Creator: HOCQUET Florian, Landza HOUDI
Date: 30/09/2022
Version: 1.0

Purpose:
"""

# IMPORT: deep learning
import torch
import torch.nn as nn

from torch.nn import MSELoss


class RMSE(nn.Module):
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self):
        super(RMSE, self).__init__()

    def forward(self, prediction, ground_truth):
        """
        Apply loss computation.

        Parameters:
        - prediction (torch.Tensor): the prediction.
        - ground_truth (torch.Tensor): the ground truth.
        """
        mse = MSELoss().to(torch.device(self._DEVICE))
        return torch.sqrt(mse(prediction, ground_truth))
