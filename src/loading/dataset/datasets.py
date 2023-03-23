"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *
from tqdm import tqdm

# IMPORT: data processing
import torch

# IMPORT: project
from .dataset import DataSet


class UnsupervisedDataSet(DataSet):
    """
    Represents a dataset for unsupervised deep learning problem.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour
        data_info : Dict[str, int]
            information about the data within the dataset
        _inputs : List[Union[str, torch.Tensor]]
            input tensors
        _file_loader : FileLoader
            components allowing to load file from path
        _data_chopper : DataChopper
            components allowing to chop

    Methods
    ----------
        _load_dataset
            Loads dataset's data from file paths
        _get_data : torch.Tensor
            Gets data from dataset
        _verify_shape
            Verifies the tensor's shape according to the desired dimension
        _adjust_shape : torch.Tensor
            Adjusts the tensor's shape according to the desired dimension
        _collect_data_info : Dict[str, int]
            Collects information about the tensor
    """

    def __init__(self, params: Dict[str, Any], inputs: List[str]):
        """
        Instantiates a DataSet.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour
            inputs : List[str]
                input tensors' paths
        """
        # Mother Class
        super(UnsupervisedDataSet, self).__init__(params, inputs)

        # Attributes
        self.data_info: Dict[str, int] = self._collect_data_info(self.__getitem__(0))

        if not self._params["lazy_loading"]:
            self._load_dataset()

    def _load_dataset(self):
        """ Loads dataset's data from file paths. """
        for idx, file_path in enumerate(tqdm(self._inputs, desc="Loading the data in RAM.")):
            self._inputs[idx] = self._get_data(self._inputs[idx])

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Parameters
        ----------
            idx : int
                index of the item to get within the dataset

        Returns
        ----------
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
                either a single tensor or a tuple of tensors
        """
        # 2D data
        if self._params["input_dim"] == 2:
            return self._get_data(self._inputs[idx])

        # 3D data
        elif self._params["input_dim"] == 3:
            return self._data_chopper(self._get_data(self._inputs[idx]))


class SupervisedDataSet(DataSet):
    """
    Represents a dataset for supervised deep learning problem.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour
        data_info : Dict[str, int]
            information about the data within the dataset
        _inputs : List[Union[str, torch.Tensor]]
            input tensors
        _targets : List[Union[str, torch.Tensor]]
            target tensors
        _file_loader : FileLoader
            components allowing to load file from path
        _data_chopper : DataChopper
            components allowing to chop

    Methods
    ----------
        _load_dataset
            Loads dataset's data from file paths
        _get_data : torch.Tensor
            Gets data from dataset
        _verify_shape
            Verifies the tensor's shape according to the desired dimension
        _adjust_shape : torch.Tensor
            Adjusts the tensor's shape according to the desired dimension
        _collect_data_info : Dict[str, int]
            Collects information about the tensor
    """

    def __init__(self, params: Dict[str, Any], inputs: List[str], targets: List[str] = None):
        """
        Instantiates a DataSet.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour
            inputs : List[str]
                input tensors' paths
            targets : List[str]
                target tensors' paths
        """
        # Mother Class
        super(SupervisedDataSet, self).__init__(params, inputs)

        # Attributes
        self._targets: List[Union[str, torch.Tensor]] = targets
        self.data_info: Dict[str, int] = self._collect_data_info(self.__getitem__(0)[0])

        if not self._params["lazy_loading"]:
            self._load_dataset()

    def _load_dataset(self):
        """ Loads dataset's data from file paths. """
        for idx, file_path in enumerate(tqdm(self._inputs, desc="Loading the data in RAM.")):
            self._inputs[idx] = self._get_data(self._inputs[idx])
            self._targets[idx] = self._get_data(self._targets[idx])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
            idx : int
                index of the item to get within the dataset

        Returns
        ----------
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
                either a single tensor or a tuple of tensors
        """
        # 2D data
        if self._params["input_dim"] == 2:
            return self._get_data(self._inputs[idx]), self._get_data(self._targets[idx])

        # 3D data
        elif self._params["input_dim"] == 3:
            return self._data_chopper(
                self._get_data(self._inputs[idx]),
                self._get_data(self._targets[idx])
            )
