"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: deep learning
import torch
from monai.networks.nets import BasicUNet, AttentionUnet, UNETR, SwinUNETR

# IMPORT: project
from .model import Model


class UNet(Model, BasicUNet):
    _DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def __init__(self, input_shape, target_shape, weights_path=None):
        # Model mother class
        Model.__init__(self, input_shape, target_shape, weights_path=None)

        # BasicUNet mother class
        BasicUNet.__init__(
            self,
            spatial_dims=self._params["spatial_dims"],
            in_channels=self._params["in_channels"], out_channels=self._params["out_channels"],
            dropout=0.2,
        )

        self.to(self._DEVICE)
        if weights_path is not None:
            self._model.load_state_dict(torch.load(weights_path))

    def __str__(self):
        return "UNet"


class AttentionUNet(Model, AttentionUnet):
    _DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def __init__(self, input_shape, target_shape, weights_path=None):
        # Model mother class
        Model.__init__(self, input_shape, target_shape, weights_path=None)

        # AttentionUnet mother class
        AttentionUnet.__init__(
            self,
            spatial_dims=self._params["spatial_dims"], channels=(8, 16, 32),
            in_channels=self._params["in_channels"], out_channels=self._params["out_channels"],
            strides=(2, 2), dropout=0.2,
        )

        self.to(self._DEVICE)
        if weights_path is not None:
            self._model.load_state_dict(torch.load(weights_path))

    def __str__(self):
        return "AttentionUNet"


class Transformer(Model, UNETR):
    _DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def __init__(self, input_shape, target_shape, weights_path=None):
        # Model mother class
        Model.__init__(self, input_shape, target_shape, weights_path=None)

        # UNETR mother class
        UNETR.__init__(
            self,
            img_size=self._params["img_size"], spatial_dims=self._params["spatial_dims"],
            in_channels=self._params["in_channels"], out_channels=self._params["out_channels"],
            dropout_rate=0.2
        )

        self.to(self._DEVICE)
        if weights_path is not None:
            self._model.load_state_dict(torch.load(weights_path))

    def __str__(self):
        return "Transformer"


class SWinTransformer(Model, SwinUNETR):
    def __init__(self, input_shape, target_shape, weights_path=None):
        # Model mother class
        Model.__init__(self, input_shape, target_shape, weights_path=None)

        # SwinUNETR mother class
        SwinUNETR.__init__(
            self,
            img_size=self._params["img_size"], spatial_dims=self._params["spatial_dims"],
            in_channels=self._params["in_channels"], out_channels=self._params["out_channels"],
            drop_rate=0.2
        )

        self.to(self._DEVICE)
        if weights_path is not None:
            self._model.load_state_dict(torch.load(weights_path))

    def __str__(self):
        return "SWinTransformer"
