"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import os

# IMPORT: data loading
import torch


class DataLoader:
    def __init__(self):
        self._file_extensions = None

    def _verify_path(self, path: str):
        if self._file_extensions is None:
            raise TypeError("The extensions' list has not been defined.")

        if not os.path.basename(path).split(".")[1] in self._file_extensions:
            raise ValueError("The file extension isn't supported.")

        return True

    def _load(self, path: str) -> torch.Tensor:
        raise NotImplementedError("Mother class.")

    def __call__(self, path):
        self._verify_path(path)
        return self._load(path)
