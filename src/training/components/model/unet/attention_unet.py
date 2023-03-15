"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: deep learning
import torch
from monai.networks.nets import AttentionUnet as MonaiAttentionUnet


class AttentionUNet(MonaiAttentionUnet):
    def __init__(self, weights_path, spatial_dims, in_channels, out_channels):
        # Mother class
        super(AttentionUNet, self).__init__(
            spatial_dims=spatial_dims, channels=(8, 16, 32),
            in_channels=in_channels, out_channels=out_channels,
            strides=(2, 2), dropout=0.2
        )

        # Attributes
        self._name = "Attention_UNet"

        if weights_path is not None:
            self.load_state_dict(torch.load(weights_path))

    @property
    def name(self):
        return self._name


class AttentionUNet2D(AttentionUNet):
    def __init__(self, weights_path, out_channels):
        # Mother class
        super(AttentionUNet2D, self).__init__(
            weights_path,
            spatial_dims=2, in_channels=1, out_channels=out_channels
        )

        # Attributes
        self._name += "_2d"


class AttentionUNet25D(AttentionUNet):
    def __init__(self, weights_path, in_channels, out_channels):
        # Mother class
        super(AttentionUNet25D, self).__init__(
            weights_path,
            spatial_dims=2, in_channels=in_channels, out_channels=out_channels
        )

        # Attributes
        self._name += "_25d"


class AttentionUNet3D(AttentionUNet):
    def __init__(self, weights_path, out_channels):
        # Mother class
        super(AttentionUNet3D, self).__init__(
            weights_path,
            spatial_dims=3, in_channels=1, out_channels=out_channels
        )

        # Attributes
        self._name += "_3d"
