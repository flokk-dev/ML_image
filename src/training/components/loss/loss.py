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


class Loss:
    """
    Represents a general loss function, that will be derived depending on the use case.

    Attributes
    ----------
        _loss : torch.nn.Module
            loss function to apply.
        _behaviour: str
            loss' behaviour
        _params : Dict[str, int]
            parameters needed to adjust the loss behaviour
    """

    _DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def __init__(self, params: Dict[str, int]):
        """
        Instantiates a Loss.

        Parameters
        ----------
            params : Dict[str, int]
                parameters needed to adjust the loss behaviour
        """
        # Attributes
        self._loss: torch.nn.Module = None
        self._behaviour: str = None

        self._params: Dict[str, int] = params

    @property
    def behaviour(self) -> str:
        """
        Returns the loss' behaviour.

        Returns
        ----------
            str
                loss' behaviour
        """
        return self._behaviour

    def __call__(self, prediction_batch: torch.Tensor, target_batch: torch.Tensor = None) \
            -> torch.Tensor:
        """
        Parameters
        ----------
            prediction_batch : torch.Tensor
                batch of predicted tensors
            target_batch : torch.Tensor
                batch of target tensors

        Raises
        ----------
            NotImplementedError
                function isn't implemented yet
        """
        raise NotImplementedError()
