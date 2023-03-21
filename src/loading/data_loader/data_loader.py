"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import typing

# IMPORT: data loading
from torch.utils.data import DataLoader as TorchDataLoader

# IMPORT: project
from src.loading.dataset.dataset import DataSet


class DataLoader(TorchDataLoader):
    def __init__(
        self,
        params: typing.Dict[str, typing.Any],
        dataset: DataSet
    ):
        """
        pass.
        """
        # Mother Class
        super(DataLoader, self).__init__(
            dataset,
            batch_size=params["batch_size"],
            shuffle=True, collate_fn=self._collate_fn
        )

        # Attributes
        self._params: dict = params
        self.data_info: dict = dataset.data_info

    def _collate_fn(
        self,
        data: typing.List[typing.Any]
    ) -> TorchDataLoader:
        """
        pass.
        """
        raise NotImplementedError()
