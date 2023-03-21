"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import typing

# IMPORT: project
from .unet import UNet, AttentionUNet, Transformer, SWinTransformer


class ModelManager(dict):

    def __init__(self):
        super(ModelManager, self).__init__({
            "unet": UNet,
            "attention unet": AttentionUNet,
            "transformer": Transformer,
            "swin transformer": SWinTransformer,
        })

    def __call__(
            self,
            model_id: str,
            data_info: typing.Dict[str, int],
            weights_path: str = None
    ) -> typing.Any:
        try:
            return self[model_id](data_info, weights_path)
        except KeyError:
            raise KeyError(f"The {model_id} isn't handled by the metric manager.")
