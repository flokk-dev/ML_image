"""
Creator: HOCQUET Florian, LANDZA Houdi
Date: 30/09/2022
Version: 1.0

Purpose: Represents a 2.5D U-Net.
"""

# IMPORT: deep learning
import torch
from monai.networks.nets import AttentionUnet


class AttentionUnet3D(AttentionUnet):
    """
    The class representing a 2.5D U-Net.

    Attributes:
    - name: the model's name.
    - weights_path: the model's weights dict.
    """

    def __init__(self, weights_path: str = None, in_channels: int = 1):
        """
        Initialize the UNet25D class.

        Parameters:
        - weights_path (str): the path to the model's weights dict.
        - in_channels (int): the model's number of input channels.
        """
        # Initialise the model
        super(AttentionUnet3D, self).__init__(spatial_dims=3, in_channels=in_channels,
                                              out_channels=1, channels=(8, 16, 32),
                                              strides=(2, 2), dropout=0.2)

        self.name = "UNet_25D"
        self.weights_path = weights_path

        if weights_path is not None:
            self.load_state_dict(torch.load(weights_path))
