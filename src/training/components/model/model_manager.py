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
    """ Represents a model manager. """

    def __init__(self):
        """ Instantiates a ModelManager. """
        super(ModelManager, self).__init__({
            "unet": UNet,
            "attention unet": AttentionUNet,
            "transformer": Transformer,
            "swin transformer": SWinTransformer,
        })

    def __call__(self, model_id: str, data_info: Dict[str, int], weights_path: str = None) -> Any:
        """
        Parameters
        ----------
            model_id : str
                id of the model
            data_info : Dict[str, int]
                information about the data within the dataset
            weights_path : str
                path to the model's weights

        Returns
        ----------
            torch.nn.Module
                model function associated with the model id
        """
        try:
            return self[model_id](data_info, weights_path)
        except KeyError:
            raise KeyError(f"The {model_id} isn't handled by the metric manager.")
