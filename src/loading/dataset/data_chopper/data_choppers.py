"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *

# IMPORT: data processing
import torch

# IMPORT: projet
from .data_chopper import DataChopper


class DataChopper2D(DataChopper):
    """
    Represents a data chopper outputting 2D tensors.

    Methods:
        _chop : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            Chops data to match the desired output dimension
    """

    def __init__(self):
        """ Instantiates a DataChopper2D. """
        # Mother Class
        super(DataChopper2D, self).__init__()

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

        Returns
        ----------
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
                chopped tensors
        """
        input_p: torch.Tensor = input_t.squeeze(0).permute((1, 0, 2, 3))
        if target_t is None:
            return input_p

        # TARGETS
        target_p: torch.Tensor = target_t.squeeze(0).permute((1, 0, 2, 3))
        return input_p, target_p


class DataChopper25D(DataChopper):
    """
    Represents a data chopper outputting 2.5D tensors.

    Methods:
        _chop : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            Chops data to match the desired output dimension
    """

    def __init__(self):
        """ Instantiates a DataChopper25D. """
        # Mother Class
        super(DataChopper25D, self).__init__()

        # Attributes
        self._patch_height = 5

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

        Returns
        ----------
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
                chopped tensors
        """
        input_t: torch.Tensor = input_t.squeeze(0).squeeze(0)
        input_p: torch.Tensor = input_t.unfold(0, self._patch_height, 1).permute((0, 3, 1, 2))

        if target_t is None:
            return input_p

        # TARGETS
        gap: int = self._patch_height // 2
        target_p: torch.Tensor = target_t.squeeze(0).permute((1, 0, 2, 3))[gap:-gap]

        return input_p, target_p


class DataChopper3D(DataChopper):
    """
    Represents a data chopping method outputting 3D tensors.

    Methods:
        _chop : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            Chops data to match the desired output dimension
    """

    def __init__(self):
        """ Instantiates a DataChopper3D. """
        # Mother Class
        super(DataChopper3D, self).__init__()

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
