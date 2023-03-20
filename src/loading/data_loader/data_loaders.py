"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import typing

# IMPORT: data processing
import torch
from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader

# IMPORT: project
from .data_loader import DataLoader
from src.loading.dataset.dataset import DataSet


class UnsupervisedDataLoader(DataLoader):
    def __init__(
            self,
            params: typing.Dict[str, typing.Any],
            dataset: DataSet
    ):
        """
        pass.
        """
        # Mother Class
        super(UnsupervisedDataLoader, self).__init__(params, dataset)

    def _collate_fn(
            self,
            data: typing.List[torch.Tensor]
    ) -> TorchDataLoader:
        """
        pass.
        """
        inputs: torch.Tensor = torch.cat(data, dim=0)

        return TorchDataLoader(
            TensorDataset(inputs),
            batch_size=self._params["batch_size"], shuffle=True, drop_last=True
        )


class SupervisedDataLoader(DataLoader):
    def __init__(
            self,
            params: typing.Dict[str, typing.Any],
            dataset: DataSet
    ):
        """
        pass.
        """
        # Mother Class
        super(SupervisedDataLoader, self).__init__(params, dataset)

    def _collate_fn(
            self,
            data: typing.List[typing.Tuple[torch.Tensor, torch.Tensor]]
    ) -> TorchDataLoader:
        """
        pass.
        """
        tensors: tuple = tuple(zip(*[(input_t, target_t) for input_t, target_t in data]))

        input_t: torch.Tensor = torch.cat(tensors[0], dim=0)
        target_t: torch.Tensor = torch.cat(tensors[1], dim=0)

        return TorchDataLoader(
            TensorDataset(input_t, target_t),
            batch_size=self._params["batch_size"], shuffle=True, drop_last=True
        )
