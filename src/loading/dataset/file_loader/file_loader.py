"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import os
import typing


# IMPORT: loading
import torch


class FileLoader:
    def __init__(self):
        """ Represents a general loading method, that will be derived depending on the use case. """
        self._file_extensions: typing.List[str] = list()

    def _verify_path(
            self,
            file_path: str
    ):
        """
        Verifies if file_path's extension is handled by the data loader.

        Args:
            file_path (str): path to the file

        Raises:
            ValueError: the file_path's extension isn't handled by the data loader
        """
        if not os.path.basename(file_path).split(".")[1] in self._file_extensions:
            raise ValueError("The file extension isn't supported.")

    def _load(
            self,
            file_path: str
    ) -> torch.Tensor:
        """
        Loads a file into a Tensor.

        Args:
            file_path (str): path to the file

        Returns:
            torch.Tensor: file's content as a Tensor

        Raises:
            NotImplementedError: abstract method
        """
        raise NotImplementedError("Abstract method.")

    def __call__(
            self,
            file_path: str
    ) -> torch.Tensor:
        """
        Verifies and loads a file into a Tensor.

        Args:
            file_path (str): path to the file

        Returns:
            torch.Tensor: file's content as a Tensor
        """
        self._verify_path(file_path)
        return self._load(file_path)
