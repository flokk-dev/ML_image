"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: data loading
import numpy as np
import torch
import zstd
import cv2

# IMPORT: project
from .file_loader import FileLoader


class ImageLoader(FileLoader):
    """
    Represents the image loader.

    Attributes
    ----------
        _file_extensions : str
            supported extensions

    Methods
    ----------
        _verify_path
            Verifies if file_path's extension is handled by the data loader
        _load : torch.Tensor
            Loads a file into a Tensor
    """

    def __init__(self):
        """ Instantiates an ImageLoader. """
        # Mother Class
        super(ImageLoader, self).__init__()

        # Attributes
        self._file_extensions.extend(["png", "jpg", "jpeg"])

    def _load(self, file_path: str) -> torch.Tensor:
        """
        Loads an image into a Tensor.

        Parameters
        ----------
            file_path : str
                path to the file

        Returns
        ----------
            torch.Tensor
                file's content as a Tensor
        """
        array: np.ndarray = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        return torch.from_numpy(array).type(torch.float32)


class NpyLoader(FileLoader):
    """
    Represents the npy file loader.

    Attributes
    ----------
        _file_extensions : str
            supported extensions.

    Methods
    ----------
        _verify_path
            Verifies if file_path's extension is handled by the data loader.
        _load : torch.Tensor
            Loads a file into a Tensor.
    """

    def __init__(self):
        """ Instantiates a NpyLoader. """
        # Mother Class
        super(NpyLoader, self).__init__()

        # Attributes
        self._file_extensions.extend(["npy"])

    def _load(self, file_path: str) -> torch.Tensor:
        """
        Loads a npy file's content into a Tensor.

        Parameters
        ----------
            file_path : str
                path to the file

        Returns
        ----------
            torch.Tensor
                file's content as a Tensor
        """
        array: np.ndarray = np.load(file_path).copy()
        return torch.from_numpy(array).type(torch.float32)


class NpzLoader(FileLoader):
    """
    Represents the npz file loader.

    Attributes
    ----------
        _file_extensions : str
            supported extensions.

    Methods
    ----------
        _verify_path
            Verifies if file_path's extension is handled by the data loader.
        _load : torch.Tensor
            Loads a file into a Tensor.
    """

    def __init__(self):
        """ Instantiates a NpzLoader. """
        # Mother Class
        super(NpzLoader, self).__init__()

        # Attributes
        self._file_extensions: list = ["npz"]

    def _load(self, file_path: str) -> torch.Tensor:
        """
        Loads a npz file's content into a Tensor.

        Parameters
        ----------
            file_path : str
                path to the file

        Returns
        ----------
            torch.Tensor
                file's content as a Tensor
        """
        file_content: np.NpzFile = np.load(file_path, allow_pickle=True)
        header: dict = file_content["header"][()]

        array: bytes = zstd.decompress(file_content["data"])
        array: np.ndarray = np.frombuffer(array, dtype=header["dtype"]).copy()
        array: np.ndarray = np.reshape(array, header["shape"])

        return torch.from_numpy(array).type(torch.float32)


class PtLoader(FileLoader):
    """
    Represents the pt file loader.

    Attributes
    ----------
        _file_extensions : str
            supported extensions.

    Methods
    ----------
        _verify_path
            Verifies if file_path's extension is handled by the data loader.
        _load : torch.Tensor
            Loads a file into a Tensor.
    """

    def __init__(self):
        """ Instantiates a PtLoader. """
        # Mother Class
        super(PtLoader, self).__init__()

        # Attributes
        self._file_extensions: list = ["pt"]

    def _load(self, file_path: str) -> torch.Tensor:
        """
        Loads a pt file's content into a Tensor.

        Parameters
        ----------
            file_path : str
                path to the file

        Returns
        ----------
            torch.Tensor
                file's content as a Tensor
        """
        return torch.load(file_path).type(torch.float32)
