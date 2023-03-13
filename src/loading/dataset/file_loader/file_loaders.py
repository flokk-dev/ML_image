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
        # Mother Class
        super(ImageLoader, self).__init__()

        # Attributes
        self._file_extensions = ["png", "jpg", "jpeg"]

    def _load(self, path):
        array = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        return torch.from_numpy(array).type(torch.float32)


class NpyLoader(FileLoader):
    def __init__(self):
        # Mother Class
        super(NpyLoader, self).__init__()

        # Attributes
        self._file_extensions = ["npy"]

    def _load(self, path):
        array = np.load(path).copy()
        return torch.from_numpy(array).type(torch.float32)


class NpzLoader(FileLoader):
    def __init__(self):
        # Mother Class
        super(NpzLoader, self).__init__()

        # Attributes
        self._file_extensions = ["npz"]

    def _load(self, path):
        array = np.load(path, allow_pickle=True)
        header = array["header"][()]

        array = zstd.decompress(array["data"])
        array = np.frombuffer(array, dtype=header["dtype"]).copy()
        array = np.reshape(array, header["shape"])

        return torch.from_numpy(array).type(torch.float32)


class PtLoader(FileLoader):
    def __init__(self):
        # Mother Class
        super(PtLoader, self).__init__()

        # Attributes
        self._file_extensions = ["pt"]

    def _load(self, path):
        return torch.load(path).type(torch.float32)
