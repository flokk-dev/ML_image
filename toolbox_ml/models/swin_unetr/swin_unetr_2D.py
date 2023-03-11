"""
Creator: HOCQUET Florian, LANDZA Houdi
Date: 30/09/2022
Version: 1.0

Purpose: Represents a 2D U-Net.
"""

# IMPORT: deep learning
import torch
from monai.networks.nets import SwinUNETR


class SwinUNETR2D(SwinUNETR):
    """
    The class representing a 2D U-Net.

    Attributes:
    - name: the models's name.
    - weights_path: the models's weights dict.
    """

    def __init__(self, weights_path: str = None, in_channels: int = 1):
        """
        Initialize the UNet2D class.

        Parameters:
        - weights_path (str): the path to the models's weights dict.
        - in_channels (int): the models's number of input channels.
        """
        # Initialise the models
        super(SwinUNETR2D, self).__init__(img_size=(512, 512), spatial_dims=2,
                                          in_channels=in_channels, out_channels=1,
                                          drop_rate=0.2)

        self.name = "UNet_2D"
        self.weights_path = weights_path

        if weights_path is not None:
            self.load_state_dict(torch.load(weights_path))