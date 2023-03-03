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
from .data_loader import DataLoader


class ImageLoader(DataLoader):
    def __init__(self):
        # Mother Class
        super(ImageLoader, self).__init__()

        # Attributes
        self._file_extensions = ["png", "jpg", "jpeg"]

    def _load(self, path: str):
        array = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        return torch.from_numpy(array).type(torch.float32)


class NumpyLoader(DataLoader):
    def __init__(self):
        # Mother Class
        super(NumpyLoader, self).__init__()

        # Attributes
        self._file_extensions = ["npy"]

    def _load(self, path: str):
        array = np.load(path).copy()
        return torch.from_numpy(array).type(torch.float32)

class ZSTDLoader(DataLoader):
    def __init__(self):
        # Mother Class
        super(ZSTDLoader, self).__init__()

        # Attributes
        self._file_extensions = ["npz"]

    def _load(self, path: str):
        array = np.load(path, allow_pickle=True)
        header = array["header"][()]

        array = zstd.decompress(array["data"])
        array = np.frombuffer(array, dtype=header["dtype"]).copy()
        array = np.reshape(array, header["shape"])

        return torch.from_numpy(array).type(torch.float32)


class TensorLoader(DataLoader):
    def __init__(self):
        # Mother Class
        super(TensorLoader, self).__init__()

        # Attributes
        self._file_extensions = ["pt"]

    def _load(self, path: str):
        return torch.load(path).type(torch.float32)
