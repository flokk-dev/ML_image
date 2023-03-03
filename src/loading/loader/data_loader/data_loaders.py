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
from torch.utils.data import TensorDataset, DataLoader

# IMPORT: project
from src.loading.dataset import DataSet


class LazyLoader(DataLoader):
    def __init__(self, params: dict, dataset: DataSet):
        # Mother Class
        super(LazyLoader, self).__init__(
            dataset,
            batch_size=params["batch_size"], shuffle=True
        )

    @staticmethod
    def _collate_fn(data: list) -> tuple:
        data = list(zip(*data))

        inputs = torch.cat(data[0], dim=0)
        targets = torch.cat(data[1], dim=0)

        idx = torch.randperm(inputs.shape[0])
        return inputs[idx], targets[idx]


class TensorLoader(DataLoader):
    def __init__(self, params: dict, dataset: DataSet):
        tensors = list(zip(*[batch for batch in tqdm(DataLoader(dataset))]))

        inputs = torch.cat(tensors[0]).squeeze(dim=1)
        targets = torch.cat(tensors[1]).squeeze(dim=1)

        # Mother Class
        super(TensorLoader, self).__init__(
            TensorDataset(inputs, targets),
            num_workers=params["workers"], batch_size=params["batch_size"], shuffle=True, drop_last=True
        )
