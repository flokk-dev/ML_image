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


class Loss:
    _DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def __init__(self):
        # Mother class
        super(Loss, self).__init__()

        # Attributes
        self._loss: typing.Any = None

    def __call__(
            self,
            prediction: torch.Tensor,
            target: torch.Tensor
    ) -> typing.Any:
        raise NotImplementedError()


class CompositeLoss:
    _DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def __init__(self):
        # Mother class
        super(CompositeLoss, self).__init__()

        # Attributes
        self._losses: typing.Any = None
        self._weights: typing.Any = None

    def _verify_weights(
            self,
            weights: dict
    ):
        if len(weights) != len(self._losses):
            raise ValueError(
                f"The number of weights isn't valid, "
                f"{len(self._losses)} elements are required."
            )

        if sum(weights) != 1.0:
            raise ValueError("The sum of the weights must be equal to 1.")

    def __call__(
            self,
            prediction: torch.Tensor,
            target: torch.Tensor
    ) -> typing.Any:
        raise NotImplementedError()
