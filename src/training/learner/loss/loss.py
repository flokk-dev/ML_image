"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: deep learning
import torch


class Loss:
    _DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def __init__(self):
        # Mother class
        super(Loss, self).__init__()

        # Attributes
        self._loss = None

    def __call__(self, prediction, target):
        raise NotImplementedError()


class CompositeLoss:
    _DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def __init__(self):
        # Mother class
        super(CompositeLoss, self).__init__()

        # Attributes
        self._losses = None
        self._weights = None

    def _verify_weights(self, weights):
        if len(weights) != len(self._losses):
            raise ValueError(
                f"The number of weights isn't valid, "
                f"{len(self._losses)} elements are required."
            )

        if sum(weights) != 1.0:
            raise ValueError("The sum of the weights must be equal to 1.")

    def __call__(self, prediction, target):
        raise NotImplementedError()
