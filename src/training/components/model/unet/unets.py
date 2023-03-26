"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *

# IMPORT: deep learning
import torch
from monai.networks.nets import BasicUNet, AttentionUnet, UNETR, SwinUNETR


class UNet(BasicUNet):
    """
    Represents an U-Net model.

    Attributes
    ----------
        _name : str
            the model's name
    """

    _DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def __init__(self, data_info: Dict[str, int], weights_path: str = None):
        """
        Instantiates a UNet.

        Parameters
        ----------
            data_info : Dict[str, int]
                information about the data within the dataset
            weights_path : str
                path to the model's weights
        """
        # Mother class
        super(UNet, self).__init__(
            spatial_dims=data_info["spatial_dims"],
            in_channels=data_info["in_channels"],
            out_channels=data_info["out_channels"],
            dropout=0.2,
        )
        self.to(self._DEVICE)

        # Attributes
        self._name: str = f"UNet_{data_info['spatial_dims']}D"

        if weights_path is not None:
            self.load_state_dict(torch.load(weights_path))

    def __str__(self) -> str:
        """
        Returns
        ----------
            str
                the model's name
        """
        return self._name


class AttentionUNet(AttentionUnet):
    """
    Represents an attention U-Net model.

    Attributes
    ----------
        _name : str
            the model's name
    """

    _DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def __init__(self, data_info: Dict[str, int], weights_path: str = None):
        """
        Instantiates a AttentionUNet.

        Parameters
        ----------
            data_info : Dict[str, int]
                information about the data within the dataset
            weights_path : str
                path to the model's weights
        """
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
        self._name: str = f"AttentionUNet_{data_info['spatial_dims']}D"

        if weights_path is not None:
            self.load_state_dict(torch.load(weights_path))

    def __str__(self) -> str:
        """
        Returns
        ----------
            str
                the model's name
        """
        return self._name


class Transformer(UNETR):
    """
    Represents a transformer model.

    Attributes
    ----------
        _name : str
            the model's name
    """

    _DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def __init__(self, data_info: Dict[str, int], weights_path: str = None):
        """
        Instantiates a Transformer.

        Parameters
        ----------
            data_info : Dict[str, int]
                information about the data within the dataset
            weights_path : str
                path to the model's weights
        """
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
        self._name: str = f"Transformer_{data_info['spatial_dims']}D"

        if weights_path is not None:
            self.load_state_dict(torch.load(weights_path))

    def __str__(self) -> str:
        """
        Returns
        ----------
            str
                the model's name
        """
        return self._name


class SWinTransformer(SwinUNETR):
    """
    Represents a SWin transformer model.

    Attributes
    ----------
        _name : str
            the model's name
    """

    _DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def __init__(self, data_info: Dict[str, int], weights_path: str = None):
        """
        Instantiates a SWinTransformer.

        Parameters
        ----------
            data_info : Dict[str, int]
                information about the data within the dataset
            weights_path : str
                path to the model's weights
        """
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
        self._name: str = f"SWinTransformer_{data_info['spatial_dims']}D"

        if weights_path is not None:
            self.load_state_dict(torch.load(weights_path))

    def __str__(self) -> str:
        """
        Returns
        ----------
            str
                the model's name
        """
        return self._name
