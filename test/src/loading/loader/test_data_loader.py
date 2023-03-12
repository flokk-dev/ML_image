"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import os
import torch

# IMPORT: test
import pytest

# IMPORT: project
import paths

from src.loading.loader.data_loader.data_loader import DataLoader
from src.loading.loader.data_loader import UnsupervisedDataLoader, SupervisedDataLoader

from src.loading.dataset import DataSet2D, DataSet3D


# -------------------- CONSTANT -------------------- #

DATA_PATHS = {
    "2D": os.path.join(paths.TEST_DATA_PATH, "data_dim", "2D", "tensor", "input.pt"),
    "3D": os.path.join(paths.TEST_DATA_PATH, "data_dim", "3D", "tensor", "input.pt"),
}

LENGHT_TENSOR_3D = 32


# -------------------- FIXTURES -------------------- #

def dataset_to_modify(training_type, lazy_loading, input_dim, output_dim):
    datasets = {2: DataSet2D, 3: DataSet3D}
    return datasets[input_dim](
        params={
            "training_type": training_type, "file_type": "tensor", "lazy_loading": lazy_loading,
            "input_dim": input_dim, "output_dim": output_dim
        },
        inputs=[DATA_PATHS[f"{str(input_dim)}D"] for i in range(10)],
        targets=[DATA_PATHS[f"{str(input_dim)}D"] for i in range(10)]
    )


# -------------------- DATA LOADER -------------------- #
def test_data_loader():
    dataset = dataset_to_modify(
        training_type="supervised", lazy_loading=True,
        input_dim=2, output_dim=2
    )

    data_loader = DataLoader({"batch_size": 32}, dataset)
    with pytest.raises(NotImplementedError):
        data_loader._collate_fn([])


# -------------------- SUPERVISED DATA LOADER -------------------- #

def test_unsupervised_lazy_loader_2d():
    dataset = dataset_to_modify(
        training_type="unsupervised", lazy_loading=True,
        input_dim=2, output_dim=2
    )

    # batch size --> from 10 to 20
    for batch_size in range(10, 21):
        lazy_loader = UnsupervisedDataLoader({"batch_size": batch_size}, dataset)

        for idx, inputs in enumerate(lazy_loader):
            assert isinstance(inputs, torch.Tensor)
            assert inputs.dtype == torch.float32

            if batch_size*(idx+1) <= len(lazy_loader):
                assert inputs.shape == torch.Size((batch_size, 1, 32, 32))
            else:
                batch_length = len(lazy_loader) % batch_size
                assert inputs.shape == torch.Size((batch_length, 1, 32, 32))


def test_unsupervised_lazy_loader_3d():
    dataset = dataset_to_modify(
        training_type="unsupervised", lazy_loading=True,
        input_dim=3, output_dim=2
    )

    # batch size --> from 10 to 20
    for batch_size in range(10, 21):
        lazy_loader = UnsupervisedDataLoader({"batch_size": batch_size}, dataset)

        for idx, inputs in enumerate(lazy_loader):
            assert isinstance(inputs, torch.Tensor)
            assert inputs.dtype == torch.float32

            if batch_size*(idx+1) <= len(lazy_loader):
                batch_length = LENGHT_TENSOR_3D * batch_size
                assert inputs.shape == torch.Size((batch_length, 1, 32, 32))
            else:
                batch_length = LENGHT_TENSOR_3D * (len(lazy_loader) % batch_size)
                assert inputs.shape == torch.Size((batch_length, 1, 32, 32))


# -------------------- SUPERVISED DATA LOADER -------------------- #

def test_supervised_lazy_loader_2d():
    dataset = dataset_to_modify(
        training_type="supervised", lazy_loading=True,
        input_dim=2, output_dim=2
    )

    # batch size --> from 10 to 20
    for batch_size in range(10, 21):
        lazy_loader = SupervisedDataLoader({"batch_size": batch_size}, dataset)

        for idx, (inputs, targets) in enumerate(lazy_loader):
            assert isinstance(inputs, torch.Tensor)
            assert inputs.dtype == torch.float32

            assert isinstance(targets, torch.Tensor)
            assert targets.dtype == torch.float32

            if batch_size*(idx+1) <= len(lazy_loader):
                assert inputs.shape == torch.Size((batch_size, 1, 32, 32))
                assert targets.shape == torch.Size((batch_size, 1, 32, 32))
            else:
                batch_length = len(lazy_loader) % batch_size

                assert inputs.shape == torch.Size((batch_length, 1, 32, 32))
                assert targets.shape == torch.Size((batch_length, 1, 32, 32))


def test_supervised_lazy_loader_3d():
    dataset = dataset_to_modify(
        training_type="supervised", lazy_loading=True,
        input_dim=3, output_dim=2
    )

    # batch size --> from 10 to 20
    for batch_size in range(10, 21):
        lazy_loader = SupervisedDataLoader({"batch_size": batch_size}, dataset)

        for idx, (inputs, targets) in enumerate(lazy_loader):
            assert isinstance(inputs, torch.Tensor)
            assert inputs.dtype == torch.float32

            assert isinstance(targets, torch.Tensor)
            assert targets.dtype == torch.float32

            if batch_size*(idx+1) <= len(lazy_loader):
                batch_length = LENGHT_TENSOR_3D * batch_size

                assert inputs.shape == torch.Size((batch_length, 1, 32, 32))
                assert targets.shape == torch.Size((batch_length, 1, 32, 32))
            else:
                batch_length = LENGHT_TENSOR_3D * (len(lazy_loader) % batch_size)

                assert inputs.shape == torch.Size((batch_length, 1, 32, 32))
                assert targets.shape == torch.Size((batch_length, 1, 32, 32))


def test_supervised_tensor_loader_2d():
    dataset = dataset_to_modify(
        training_type="supervised", lazy_loading=False,
        input_dim=2, output_dim=2
    )

    # batch size --> from 10 to 20
    for batch_size in range(10, 21):
        tensor_loader = SupervisedDataLoader({"batch_size": batch_size}, dataset)

        for idx, (inputs, targets) in enumerate(tensor_loader):
            assert isinstance(inputs, torch.Tensor)
            assert inputs.dtype == torch.float32

            assert isinstance(targets, torch.Tensor)
            assert targets.dtype == torch.float32

            if batch_size*(idx+1) <= len(tensor_loader):
                assert inputs.shape == torch.Size((batch_size, 1, 32, 32))
                assert targets.shape == torch.Size((batch_size, 1, 32, 32))
            else:
                batch_length = len(tensor_loader) % batch_size

                assert inputs.shape == torch.Size((batch_length, 1, 32, 32))
                assert targets.shape == torch.Size((batch_length, 1, 32, 32))


def test_supervised_tensor_loader_3d():
    dataset = dataset_to_modify(
        training_type="supervised", lazy_loading=False,
        input_dim=3, output_dim=2
    )

    # batch size --> from 10 to 20
    for batch_size in range(10, 21):
        tensor_loader = SupervisedDataLoader({"batch_size": batch_size}, dataset)

        for idx, (inputs, targets) in enumerate(tensor_loader):
            assert isinstance(inputs, torch.Tensor)
            assert inputs.dtype == torch.float32

            assert isinstance(targets, torch.Tensor)
            assert targets.dtype == torch.float32

            if batch_size*(idx+1) <= len(tensor_loader):
                batch_length = LENGHT_TENSOR_3D * batch_size

                assert inputs.shape == torch.Size((batch_length, 1, 32, 32))
                assert inputs.shape == torch.Size((batch_length, 1, 32, 32))
            else:
                batch_length = LENGHT_TENSOR_3D * (len(tensor_loader) % batch_size)

                assert targets.shape == torch.Size((batch_length, 1, 32, 32))
                assert targets.shape == torch.Size((batch_length, 1, 32, 32))
