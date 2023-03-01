"""
Creator: HOCQUET Florian, Landza HOUDI
Date: 30/09/2022
Version: 1.0

Purpose:
"""

# IMPORT: deep learning
import torch
import torch.nn as nn


class SAE(nn.Module):
    def __init__(self):
        super(SAE, self).__init__()

    def forward(self, prediction, ground_truth):
        """
        Apply loss computation.

        Parameters:
        - prediction (torch.Tensor): the prediction.
        - ground_truth (torch.Tensor): the ground truth.
        """

        return torch.sum(torch.abs(ground_truth - prediction))
