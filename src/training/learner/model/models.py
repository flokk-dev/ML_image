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
from monai.networks.nets import BasicUNet, AttentionUnet, UNETR, SwinUNETR


class UNet(BasicUNet):
    _DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def __init__(
            self,
            data_info: typing.Dict[str, int],
            weights_path: str = None
    ):
        # Mother class
        super(UNet, self).__init__(
            spatial_dims=data_info["spatial_dims"],
            in_channels=data_info["in_channels"],
            out_channels=data_info["out_channels"],
            dropout=0.2,
        )
        self.to(self._DEVICE)

        # Attributes
        if weights_path is not None:
            self._model.load_state_dict(torch.load(weights_path))

    def __str__(self) -> str:
        return "UNet"


class AttentionUNet(AttentionUnet):
    _DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def __init__(
            self,
            data_info: typing.Dict[str, int],
            weights_path: str = None
    ):
        # Mother class
        super(AttentionUNet, self).__init__(
            spatial_dims=data_info["spatial_dims"],
            channels=(8, 16, 32),
            in_channels=data_info["in_channels"],
            out_channels=data_info["out_channels"],
            strides=(2, 2), dropout=0.2,
        )
        self.to(self._DEVICE)

        # Attributes
        if weights_path is not None:
            self._model.load_state_dict(torch.load(weights_path))

    def __str__(self) -> str:
        return "AttentionUNet"


class Transformer(UNETR):
    _DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def __init__(
            self,
            data_info: typing.Dict[str, int],
            weights_path: str = None
    ):
        # Mother class
        super(Transformer, self).__init__(
            img_size=data_info["img_size"],
            spatial_dims=data_info["spatial_dims"],
            in_channels=data_info["in_channels"],
            out_channels=data_info["out_channels"],
            dropout_rate=0.2
        )
        self.to(self._DEVICE)

        # Attributes
        if weights_path is not None:
            self._model.load_state_dict(torch.load(weights_path))

    def __str__(self) -> str:
        return "Transformer"


class SWinTransformer(SwinUNETR):
    _DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def __init__(
            self,
            data_info: typing.Dict[str, int],
            weights_path: str = None
    ):
        # Mother class
        super(SWinTransformer, self).__init__(
            img_size=data_info["img_size"],
            spatial_dims=data_info["spatial_dims"],
            in_channels=data_info["in_channels"],
            out_channels=data_info["out_channels"],
            drop_rate=0.2
        )
        self.to(self._DEVICE)

        # Attributes
        if weights_path is not None:
            self._model.load_state_dict(torch.load(weights_path))

    def __str__(self) -> str:
        return "SWinTransformer"
