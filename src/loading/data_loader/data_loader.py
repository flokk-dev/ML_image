"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *

# IMPORT: data loading
import torch
from torch.utils.data import DataLoader as TorchDataLoader

# IMPORT: project
from src.loading.dataset.dataset import DataSet


class DataLoader(TorchDataLoader):
    """
    Represents a general data loader, that will be derived depending on the use case.

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
        Instantiates a DataLoader.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour
            dataset: DataSet
                dataset containing file paths
        """
        # Mother Class
        super(DataLoader, self).__init__(
            dataset,
            batch_size=params["batch_size"],
            shuffle=True, collate_fn=self._collate_fn
        )

        # Attributes
        self._params: Dict[str, Any] = params
        self.data_info: Dict[str, int] = dataset.data_info

    def _collate_fn(self, data: List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]) \
            -> TorchDataLoader:
        """
        Defines the data loader's behaviour when getting data.

        Parameters
        ----------
            data : List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]
                list containing the recovered data

        Raises
        ----------
            NotImplementedError
                function isn't implemented yet
        """
        raise NotImplementedError()
