"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils

# IMPORT: deep learning
import torch


class Trainer:
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, params):
        # Attributes
        self._params = params

    def _train(self):
        raise NotImplementedError()

    def _train_epoch(self, step):
        raise NotImplementedError()

    def _train_block(self, inputs, targets, batch_idx, step):
        raise NotImplementedError()
