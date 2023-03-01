"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: data loading
import numpy as np
import skimage
import torch
import zstd

# IMPORT: projet
from .data_loader import DataLoader


class ImageLoader(DataLoader):
    def __init__(self):
        # Mother Class
        super(ImageLoader, self).__init__()

    def _load(self, path):
        array = skimage.io.imread(path).copy()
        return torch.from_numpy(array).type(torch.float32)


class NumpyLoader(DataLoader):
    def __init__(self):
        # Mother Class
        super(NumpyLoader, self).__init__()

    def _load(self, path):
        array = np.load(path).copy()
        return torch.from_numpy(array).type(torch.float32)


class ZSTDLoader(DataLoader):
    def __init__(self):
        # Mother Class
        super(ZSTDLoader, self).__init__()

    def _load(self, path):
        volume = np.load(path, allow_pickle=True)
        header = volume["header"][()]

        volume = zstd.decompress(volume["data"])
        volume = np.frombuffer(volume, dtype=np.float32).copy()
        volume = np.reshape(volume, header["shape"])

        return torch.from_numpy(volume).type(torch.float32)


class TensorLoader(DataLoader):
    def __init__(self):
        # Mother Class
        super(TensorLoader, self).__init__()

    def _load(self, path):
        return torch.load(path).type(torch.float32)
