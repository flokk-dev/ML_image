"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: dataset loading
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
        self._file_extensions = ["png", "jpg", "jpeg"]

    def _load(self, file_path):
        """
        Loads an image into a Tensor.

        Args:
            file_path (str): path to the file

        Returns:
            torch.Tensor: image as a Tensor
        """
        array = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        return torch.from_numpy(array).type(torch.float32)


class NpyLoader(FileLoader):
    def __init__(self):
        """ Represents the npy file loading method. """
        # Mother Class
        super(NpyLoader, self).__init__()

        # Attributes
        self._file_extensions = ["npy"]

    def _load(self, file_path):
        """
        Loads a npy file's content into a Tensor.

        Args:
            file_path (str): path to the Numpy file

        Returns:
            torch.Tensor: npy file's content as a Tensor
        """
        array = np.load(file_path).copy()
        return torch.from_numpy(array).type(torch.float32)


class NpzLoader(FileLoader):
    def __init__(self):
        """ Represents the npz file loading method. """
        # Mother Class
        super(NpzLoader, self).__init__()

        # Attributes
        self._file_extensions = ["npz"]

    def _load(self, file_path):
        """
        Loads a npz file's content into a Tensor.

        Args:
            file_path (str): file to the path

        Returns:
            torch.Tensor: npz file's content as a Tensor
        """
        array = np.load(file_path, allow_pickle=True)
        header = array["header"][()]

        array = zstd.decompress(array["data"])
        array = np.frombuffer(array, dtype=header["dtype"]).copy()
        array = np.reshape(array, header["shape"])

        return torch.from_numpy(array).type(torch.float32)


class PtLoader(FileLoader):
    def __init__(self):
        """ Represents the pt file loading method. """
        # Mother Class
        super(PtLoader, self).__init__()

        # Attributes
        self._file_extensions = ["pt"]

    def _load(self, file_path):
        """
        Loads a pt file's content into a Tensor.

        Args:
            file_path (str): path to the file

        Returns:
            torch.Tensor: pt file's content as a Tensor
        """
        return torch.load(file_path).type(torch.float32)