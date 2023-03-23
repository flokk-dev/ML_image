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
from torch.utils.data import Dataset

# IMPORT: project
from .file_loader import FileLoader, ImageLoader, NpyLoader, NpzLoader, PtLoader
from .data_chopper import DataChopper, DataChopper2D, DataChopper25D, DataChopper3D


class DataSet(Dataset):
    """
    Represents a general dataset, that will be derived depending on the use case.

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

    _FILE_LOADERS: Dict[str, Any] = {
        "image": ImageLoader, "numpy": NpyLoader, "zstd": NpzLoader, "tensor": PtLoader
    }
    _DATA_CHOPPERS: Dict[int, Any] = {
        2: DataChopper2D, 2.5: DataChopper25D, 3: DataChopper3D
    }

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
        super(DataSet, self).__init__()

        # Attributes
        self._params: Dict[str, Any] = params
        self.data_info: Dict[str, int] = dict()

        self._inputs: List[Union[str, torch.Tensor]] = inputs

        # Components
        self._file_loader: FileLoader = self._FILE_LOADERS[params["file_type"]]()
        self._data_chopper: DataChopper = self._DATA_CHOPPERS[params["output_dim"]]()

    def _get_data(self, tensor: Union[str, torch.Tensor]) -> torch.Tensor:
        """
        Gets data from dataset.

        Parameters
        ----------
            tensor : Union[str, torch.Tensor]
                data to get, either a file's path or a tensor

        Returns
        ----------
            torch.Tensor
                either the loaded tensor or the tensor itself
        """
        if isinstance(tensor, torch.Tensor):
            return tensor

        # LOAD
        tensor: torch.Tensor = self._file_loader(tensor)

        # VERIFY SHAPE
        self._verify_shape(tensor)

        # ADJUST SHAPE
        return self._adjust_shape(tensor)

    def _verify_shape(self, tensor: torch.Tensor):
        """
        Verifies the tensor's shape according to the desired dimension.

        Parameters
        ----------
            tensor : torch.Tensor
                tensor whose shape is to be verified

        Raises
        ----------
            ValueError
                tensor's shape isn't valid
        """
        dim: int = self._params["input_dim"]

        # IF too much dimensions
        if not dim <= len(tensor.shape) <= dim + 2:
            raise ValueError(f"The tensor's shape isn't valid: {tensor.shape}")

        # IF not 2d tensor a priori
        if torch.sum((torch.Tensor(tuple(tensor.shape)) > 5)) > dim:
            raise ValueError(f"The tensor's shape isn't valid: {tensor.shape}")

        # IF not 2d tensor a priori
        if len(tensor.shape) > dim:
            if torch.sum((torch.Tensor(tuple(tensor.shape)) > 1)) >= dim + 2:
                raise ValueError(f"The tensor's shape isn't valid: {tensor.shape}")

    def _adjust_shape(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Adjusts the tensor's shape according to the desired dimension.

        Parameters
        ----------
            tensor : torch.Tensor
                tensor whose shape is to be adjusted

        Returns
        ----------
            torch.Tensor
                adjusted tensor
        """
        dim: int = self._params["input_dim"]

        #
        if len(tensor.shape) == dim + 2:
            if tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)

        #
        if len(tensor.shape) == dim + 1 and tensor.shape[-1] == min(tensor.shape):
            dims_order = (dim, *(i for i in range(dim)))
            tensor = torch.permute(tensor, dims_order)

        #
        elif len(tensor.shape) == dim:
            tensor = tensor.unsqueeze(0)

        return tensor.unsqueeze(0)

    def _collect_data_info(self, tensor: torch.Tensor) -> Dict[str, int]:
        """
        Collects information about the tensor.

        Parameters
        ----------
            tensor : torch.Tensor
                tensor to get information from

        Returns
        ----------
            Dict[str, int]
                information about the tensor
        """
        return {
            "spatial_dims": len(tensor.shape) - 2,
            "img_size": tuple(tensor.shape[2:]),
            "in_channels": tensor.shape[1],
            "out_channels": self._params["out_channels"]
        }

    def _load_dataset(self):
        """
        Loads dataset's data from file paths.

        Raises
        ----------
            NotImplementedError
                function isn't implemented yet
        """
        raise NotImplementedError()

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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
        raise NotImplementedError()

    def __len__(self) -> int:
        """
        Returns
        ----------
            int
                dataset's length
        """
        return len(self._inputs)
