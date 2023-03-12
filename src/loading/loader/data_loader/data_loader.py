"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: dataset loading
from torch.utils.data import DataLoader as TorchDataLoader

# IMPORT: project
from src.loading.dataset.dataset import DataSet


class DataLoader(TorchDataLoader):
    def __init__(self, params: dict, dataset: DataSet):
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
