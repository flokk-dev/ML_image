"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: deep learning
import torch
import torch.nn as nn


class Metric(nn.Module):
    _DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def __init__(self):
        # Mother class
        super(Metric, self).__init__()

        # Attributes
        self._metric = None

    def __call__(self, prediction, target):
        raise NotImplementedError()