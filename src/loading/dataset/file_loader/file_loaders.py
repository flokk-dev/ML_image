"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: loading
import cv2

import numpy as np
import torch
import zstd

# IMPORT: projet
from .file_loader import FileLoader


class ImageLoader(FileLoader):
    def __init__(self):
        """ Represents the image loading method. """
        # Mother Class
        super(ImageLoader, self).__init__()

        # Attributes
        self._file_extensions: list = ["png", "jpg", "jpeg"]

    def _load(self, file_path: str) -> torch.Tensor:
        """
        Loads an image into a Tensor.

        Args:
            file_path (str): path to the file

        Returns:
            torch.Tensor: image as a Tensor
        """
        array: np.ndarray = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        return torch.from_numpy(array).type(torch.float32)


class NpyLoader(FileLoader):
    def __init__(self):
        """ Represents the npy file loading method. """
        # Mother Class
        super(NpyLoader, self).__init__()

        # Attributes
        self._file_extensions: list = ["npy"]

    def _load(self, file_path: str) -> torch.Tensor:
        """
        Loads a npy file's content into a Tensor.

        Args:
            file_path (str): path to the Numpy file

        Returns:
            torch.Tensor: npy file's content as a Tensor
        """
        array: np.ndarray = np.load(file_path).copy()
        return torch.from_numpy(array).type(torch.float32)


class NpzLoader(FileLoader):
    def __init__(self):
        """ Represents the npz file loading method. """
        # Mother Class
        super(NpzLoader, self).__init__()

        # Attributes
        self._file_extensions: list = ["npz"]

    def _load(self, file_path: str) -> torch.Tensor:
        """
        Loads a npz file's content into a Tensor.

        Args:
            file_path (str): file to the path

        Returns:
            torch.Tensor: npz file's content as a Tensor
        """
        file_content: np.NpzFile = np.load(file_path, allow_pickle=True)
        header: dict = file_content["header"][()]

        array: bytes = zstd.decompress(file_content["data"])
        array: np.ndarray = np.frombuffer(array, dtype=header["dtype"]).copy()
        array: np.ndarray = np.reshape(array, header["shape"])

        return torch.from_numpy(array).type(torch.float32)


class PtLoader(FileLoader):
    def __init__(self):
        """ Represents the pt file loading method. """
        # Mother Class
        super(PtLoader, self).__init__()

        # Attributes
        self._file_extensions: list = ["pt"]

    def _load(self, file_path: str) -> torch.Tensor:
        """
        Loads a pt file's content into a Tensor.

        Args:
            file_path (str): path to the file

        Returns:
            torch.Tensor: pt file's content as a Tensor
        """
        return torch.load(file_path).type(torch.float32)
