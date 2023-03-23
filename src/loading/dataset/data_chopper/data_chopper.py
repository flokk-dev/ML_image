"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *
import torch


class DataChopper:
    """
    Represents a general data chopper, that will be derived depending on the use case.

    Methods
    ----------
        _chop : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            Chops data to match the desired output dimension
    """

    def __init__(self):
        """ Instantiates a DataChopper. """
        pass

    def _chop(self, input_t: torch.Tensor, target_t: torch.Tensor = None) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Chops data to match the desired output dimension.

        Parameters
        ----------
            input_t : torch.Tensor
                input tensor
            target_t : torch.Tensor
                target tensor

        Raises
        ----------
            NotImplementedError
                function isn't implemented yet
        """
        raise NotImplementedError()

    def __call__(self, input_t: torch.Tensor, target_t: torch.Tensor = None) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
            input_t : torch.Tensor
                input tensor
            target_t : torch.Tensor
                target tensor

        Returns
        ----------
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
                chopped tensors
        """
        return self._chop(input_t, target_t)
