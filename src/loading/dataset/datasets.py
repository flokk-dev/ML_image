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

# IMPORT: project
from .dataset import DataSet


class UnsupervisedDataSet(DataSet):
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
        super(UnsupervisedDataSet, self).__init__(params, input_paths, target_paths)
        self.data_info: typing.Dict[str, int] = self._collect_data_info()

        # Attributes
        if not self._params["lazy_loading"]:
            self._load_dataset()

    def _collect_data_info(self) -> typing.Dict[str, int]:
        """
        pass.
        """
        input_t: torch.Tensor = self.__getitem__(0)
        return {
            "spatial_dims": len(input_t.shape) - 2,
            "img_size": tuple(input_t.shape[2:]),
            "in_channels": input_t.shape[1],
            "out_channels": 1
        }

    def __getitem__(
            self,
            idx: int
    ) -> torch.Tensor:
        """
        pass.
        """
        # 2D data
        if self._params["input_dim"] == 2:
            return self._get_data(self._inputs[idx])

        # 3D data
        elif self._params["input_dim"] == 3:
            return self._data_chopper(self._get_data(self._inputs[idx]))


class SupervisedDataSet(DataSet):
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
        super(SupervisedDataSet, self).__init__(params, input_paths, target_paths)
        self.data_info: typing.Dict[str, int] = self._collect_data_info()

        # Attributes
        if not self._params["lazy_loading"]:
            self._load_dataset()

    def _collect_data_info(self) -> typing.Dict[str, int]:
        """
        pass.
        """
        input_t: torch.Tensor
        target_t: torch.Tensor

        input_t, target_t = self.__getitem__(0)
        return {
            "spatial_dims": len(input_t.shape) - 2,
            "img_size": tuple(input_t.shape[2:]),
            "in_channels": input_t.shape[1],
            "out_channels": target_t.shape[1]
        }

    def __getitem__(
            self,
            idx: int
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        pass.
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
