"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *
import os

# IMPORT: data loading
import torch


class FileLoader:
    """
    Represents a general loader, that will be derived depending on the use case.

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
        """ Instantiates a FileLoader. """
        self._file_extensions: List[str] = list()

    def _verify_path(self, file_path: str):
        """
        Verifies if file_path's extension is handled by the data loader.

        Parameters
        ----------
            file_path : str
                path to the file

        Raises
        ----------
            ValueError
                file_path's extension isn't handled by the data loader
        """
        if not os.path.basename(file_path).split(".")[1] in self._file_extensions:
            raise ValueError("The file extension isn't supported.")

    def _load(self, file_path: str) -> torch.Tensor:
        """
        Loads a file into a Tensor.

        Parameters
        ----------
            file_path : str
                path to the file

        Raises
        ----------
            NotImplementedError
                abstract method
        """
        raise NotImplementedError("Abstract method.")

    def __call__(self, file_path: str) -> torch.Tensor:
        """
        Parameters
        ----------
            file_path : str
                path to the file

        Returns
        ----------
            torch.Tensor
                file's content as a Tensor
        """
        self._verify_path(file_path)
        return self._load(file_path)
