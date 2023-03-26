"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *
import torch

# IMPORT: project
from .unet import UNet, AttentionUNet, Transformer, SWinTransformer


class ModelManager(dict):
    """
    Represents a model manager.

    Attributes
    ----------
        _data_info : Dict[str, int]
            information about the data within the dataset
    """

    def __init__(self, data_info: Dict[str, int]):
        """
        Instantiates a ModelManager.

        data_info : Dict[str, int]
            information about the data within the dataset
        """
        super(ModelManager, self).__init__({
            "unet": UNet,
            "attention unet": AttentionUNet,
            "transformer": Transformer,
            "swin transformer": SWinTransformer,
        })

        # Attributes
        self._data_info: Dict[str, int] = data_info

    def __call__(self, model_id: str, weights_path: str = None) -> Any:
        """
        Parameters
        ----------
            model_id : str
                id of the model
            weights_path : str
                path to the model's weights

        Returns
        ----------
            torch.nn.Module
                model function associated with the model id

        Raises
        ----------
            KeyError
                loss id isn't handled by the loss manager
        """
        try:
            return self[model_id](self._data_info, weights_path)
        except KeyError:
            raise KeyError(f"The {model_id} isn't handled by the metric manager.")
