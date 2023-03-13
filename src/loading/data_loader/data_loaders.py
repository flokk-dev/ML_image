"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: data processing
import torch

# IMPORT: project
from .data_loader import DataLoader
from src.loading.dataset.dataset import DataSet


class UnsupervisedDataLoader(DataLoader):
    def __init__(self, params: dict, dataset: DataSet):
        # Mother Class
        super(UnsupervisedDataLoader, self).__init__(params, dataset)

    @staticmethod
    def _collate_fn(data: list) -> tuple:
        inputs = torch.cat(data, dim=0)

        idx = torch.randperm(inputs.shape[0])
        return inputs[idx]


class SupervisedDataLoader(DataLoader):
    def __init__(self, params: dict, dataset: DataSet):
        # Mother Class
        super(SupervisedDataLoader, self).__init__(params, dataset)

    @staticmethod
    def _collate_fn(data: list) -> tuple:
        inputs, targets = list(), list()
        for input_tensor, target_tensor in data:
            inputs.append(input_tensor)
            targets.append(target_tensor)

        inputs, targets = torch.cat(inputs, dim=0), torch.cat(targets, dim=0)

        idx = torch.randperm(inputs.shape[0])
        return inputs[idx], targets[idx]
