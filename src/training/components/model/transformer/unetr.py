"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: deep learning
import torch
from monai.networks.nets import UNETR


class UNetR(UNETR):
    def __init__(self, weights_path, img_size, spatial_dims, in_channels, out_channels):
        # Mother class
        super(UNetR, self).__init__(
            img_size=img_size, spatial_dims=spatial_dims,
            in_channels=in_channels, out_channels=out_channels,
            dropout_rate=0.2
        )

        # Attributes
        self._name = "SWin_UNetR"

        if weights_path is not None:
            self.load_state_dict(torch.load(weights_path))

    @property
    def name(self):
        return self._name


class UNetR2D(UNetR):
    def __init__(self, weights_path, img_size, out_channels):
        # Mother class
        super(UNetR2D, self).__init__(
            weights_path,
            img_size=img_size, spatial_dims=2, in_channels=1, out_channels=out_channels
        )

        # Attributes
        self._name += "_2d"


class UNetR25D(UNetR):
    def __init__(self, weights_path, img_size, in_channels, out_channels):
        # Mother class
        super(UNetR25D, self).__init__(
            weights_path,
            img_size=img_size, spatial_dims=2, in_channels=in_channels, out_channels=out_channels
        )

        # Attributes
        self._name += "_25d"


class UNetR3D(UNetR):
    def __init__(self, weights_path, img_size, out_channels):
        # Mother class
        super(UNetR3D, self).__init__(
            weights_path,
            img_size=img_size, spatial_dims=3, in_channels=1, out_channels=out_channels
        )

        # Attributes
        self._name += "_3d"
