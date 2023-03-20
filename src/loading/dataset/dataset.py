"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""
import typing

# IMPORT: utils
from tqdm import tqdm

# IMPORT: dataset loading
import torch
from torch.utils.data import Dataset

# IMPORT: project
from .file_loader import FileLoader, ImageLoader, NpyLoader, NpzLoader, PtLoader
from .data_chopper import DataChopper, DataChopper2D, DataChopper25D, DataChopper3D


class DataSet(Dataset):
    _FILE_LOADERS: typing.Dict[str, typing.Any] = {
        "image": ImageLoader, "numpy": NpyLoader, "zstd": NpzLoader, "tensor": PtLoader
    }
    _DATA_CHOPPERS: typing.Dict[int, typing.Any] = {
        2: DataChopper2D, 2.5: DataChopper25D, 3: DataChopper3D
    }

    def __init__(
            self,
            params: typing.Dict[str, typing.Any],
            input_paths: typing.List[str],
            target_paths: typing.List[str] = None
    ):
        """
        pass.
        """
        # Mother Class
        super(DataSet, self).__init__()

        # Attributes
        self._params: typing.Dict[str, typing.Any] = params
        self.data_info: typing.Dict[str, int] = dict()

        self._inputs: typing.List[str, torch.Tensor] = input_paths
        self._targets: typing.List[str, torch.Tensor] = target_paths

        # Components
        self._file_loader: FileLoader = self._FILE_LOADERS[params["file_type"]]()
        self._data_chopper: DataChopper = self._DATA_CHOPPERS[params["output_dim"]]()

    def _load_dataset(self):
        """
        pass.
        """
        for idx, file_path in enumerate(tqdm(self._inputs, desc="Loading the data in RAM.")):
            self._inputs[idx] = self._get_data(self._inputs[idx])

            if self._targets is not None:
                self._targets[idx] = self._get_data(self._targets[idx])

    def _get_data(
            self,
            tensor: [str, torch.Tensor]
    ) -> torch.Tensor:
        """
        pass.
        """
        if isinstance(tensor, torch.Tensor):
            return tensor

        # LOAD
        tensor: torch.Tensor = self._file_loader(tensor)

        # VERIFY SHAPE
        self._verify_shape(tensor)

        # ADJUST SHAPE
        return self._adjust_shape(tensor)

    def _verify_shape(
            self,
            tensor: torch.Tensor
    ):
        """
        pass.
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

    def _adjust_shape(
            self,
            tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        pass.
        """
        dim: int = self._params["input_dim"]

        #
        if len(tensor.shape) == dim+2:
            if tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)

        #
        if len(tensor.shape) == dim+1 and tensor.shape[-1] == min(tensor.shape):
            dims_order = (dim, *(i for i in range(dim)))
            tensor = torch.permute(tensor, dims_order)

        #
        elif len(tensor.shape) == dim:
            tensor = tensor.unsqueeze(0)

        return tensor.unsqueeze(0)

    def _collect_data_info(self) -> typing.Any:
        """
        pass.
        """
        raise NotImplementedError()

    def __getitem__(
            self,
            idx: int
    ) -> typing.Any:
        """
        pass.
        """
        raise NotImplementedError()

    def __len__(self) -> int:
        """
        pass.
        """
        return len(self._inputs)
