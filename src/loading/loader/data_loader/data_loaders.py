"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from tqdm import tqdm

# IMPORT: dataset loading
import torch
from torch.utils.data import TensorDataset

# IMPORT: project
from .data_loader import DataLoader
from src.loading.dataset import DataSet


class LazyLoader(DataLoader):
    def __init__(self, params: dict, dataset: DataSet):
        # Mother Class
        super(LazyLoader, self).__init__(params, dataset)

        # Attributes
        self._length = len(dataset)

    @staticmethod
    def _collate_fn(data: list) -> tuple:
        inputs, targets = list(), list()
        for input_tensor, target_tensor in data:
            inputs.append(input_tensor)
            targets.append(target_tensor)

        inputs, targets = torch.cat(inputs, dim=0), torch.cat(targets, dim=0)

        idx = torch.randperm(inputs.shape[0])
        return inputs[idx], targets[idx]


class TensorLoader(DataLoader):
    def __init__(self, params: dict, dataset: DataSet):
        inputs, targets = self._load_dataset(dataset)

        # Mother Class
        super(TensorLoader, self).__init__(params, TensorDataset(inputs, targets))

    @staticmethod
    def _collate_fn(data: list) -> tuple:
        return data

    @staticmethod
    def _load_dataset(dataset: DataSet):
        inputs, targets = list(), list()
        for input_tensor, target_tensor in tqdm(DataLoader(dataset)):
            inputs.append(input_tensor)
            targets.append(target_tensor)

        # inputs and targets
        return torch.cat(inputs).squeeze(dim=1), torch.cat(targets).squeeze(dim=1)
