"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: data processing
import torch
from torch.utils.data import TensorDataset, DataLoader

# IMPORT: project
from .data_loader import DataLoader
from src.loading.dataset.dataset import DataSet


class UnsupervisedDataLoader(DataLoader):
    def __init__(self, params: dict, dataset: DataSet):
        # Mother Class
        super(UnsupervisedDataLoader, self).__init__(params, dataset)

    def _collate_fn(self, data: list) -> torch.utils.data.DataLoader:
        inputs = torch.cat(data, dim=0)

        return DataLoader(
            TensorDataset(inputs),
            batch_size=self._params["batch_size"], shuffle=True, drop_last=True
        )


class SupervisedDataLoader(DataLoader):
    def __init__(self, params: dict, dataset: DataSet):
        # Mother Class
        super(SupervisedDataLoader, self).__init__(params, dataset)

    def _collate_fn(self, data: list) -> torch.utils.data.DataLoader:
        inputs, targets = list(), list()
        for input_tensor, target_tensor in data:
            inputs.append(input_tensor)
            targets.append(target_tensor)

        inputs, targets = torch.cat(inputs, dim=0), torch.cat(targets, dim=0)

        return DataLoader(
            TensorDataset(inputs, targets),
            batch_size=self._params["batch_size"], shuffle=True, drop_last=True
        )
