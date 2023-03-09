"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from tqdm import tqdm

# IMPORT: dataset loading
from torch.utils.data import DataLoader as TorchDataLoader, Dataset


class DataLoader(TorchDataLoader):
    def __init__(self, params: dict, dataset):
        # Mother Class
        super(DataLoader, self).__init__(
            dataset,
            batch_size=params["batch_size"],
            shuffle=True, collate_fn=self._collate_fn
        )

        # Attributes
        self._length = len(dataset)

    @staticmethod
    def _collate_fn(data: list) -> tuple:
        raise NotImplementedError()

    def __len__(self):
        return self._length
