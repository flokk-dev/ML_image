"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *

# IMPORT: data processing
import torch
from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader

# IMPORT: project
from .data_loader import DataLoader
from src.loading.dataset.dataset import DataSet


class UnsupervisedDataLoader(DataLoader):
    """
    Represents a data loader for unsupervised deep learning problem.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour
        data_info : Dict[str, int]
            information about the data within the dataset

    Methods
    ----------
        _collate_fn : TorchDataLoader
            Loads dataset's data from file paths
    """

    def __init__(self, params: Dict[str, Any], dataset: DataSet):
        """
        Instantiates a UnsupervisedDataLoader.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour
            dataset: DataSet
                dataset containing file paths
        """
        # Mother Class
        super(UnsupervisedDataLoader, self).__init__(params, dataset)

    def _collate_fn(self, data: List[torch.Tensor]) -> TorchDataLoader:
        """
        Defines the data loader's behaviour when getting data.

        Parameters
        ----------
            data : List[torch.Tensor]
                list containing the recovered data

        Returns
        ----------
            TorchDataLoader
                data loader containing the aggregated data
        """
        inputs: torch.Tensor = torch.cat(data, dim=0)

        return TorchDataLoader(
            TensorDataset(inputs),
            batch_size=self._params["batch_size"], shuffle=True, drop_last=True
        )


class SupervisedDataLoader(DataLoader):
    """
    Represents a data loader for unsupervised deep learning problem.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour
        data_info : Dict[str, int]
            information about the data within the dataset

    Methods
    ----------
        _collate_fn : TorchDataLoader
            Loads dataset's data from file paths
    """

    def __init__(self, params: Dict[str, Any], dataset: DataSet):
        """
        Instantiates a SupervisedDataLoader.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour
            dataset: DataSet
                dataset containing file paths
        """
        # Mother Class
        super(SupervisedDataLoader, self).__init__(params, dataset)

    def _collate_fn(self, data: List[Tuple[torch.Tensor, torch.Tensor]]) -> TorchDataLoader:
        """
        Defines the data loader's behaviour when getting data.

        Parameters
        ----------
            data : List[Tuple[torch.Tensor, torch.Tensor]]
                list containing the recovered data

        Returns
        ----------
            TorchDataLoader
                data loader containing the aggregated data
        """
        tensors: tuple = tuple(zip(*[(input_t, target_t) for input_t, target_t in data]))

        input_t: torch.Tensor = torch.cat(tensors[0], dim=0)
        target_t: torch.Tensor = torch.cat(tensors[1], dim=0)

        return TorchDataLoader(
            TensorDataset(input_t, target_t),
            batch_size=self._params["batch_size"], shuffle=True, drop_last=True
        )
