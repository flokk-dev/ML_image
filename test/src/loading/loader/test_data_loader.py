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

from src.loading.loader.data_loader import DataLoader
from src.loading.dataset import DataSet2D, DataSet3D


# -------------------- CONSTANT -------------------- #

DATA_PATHS = {
    "2D": os.path.join(paths.TEST_PATH, "data_dim", "2D", "tensor", "input.pt"),
    "3D": os.path.join(paths.TEST_PATH, "data_dim", "3D", "tensor", "input.pt"),
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


# -------------------- LAZY LOADER -------------------- #

def test_lazy_loader_2d():
    dataset = dataset_to_modify(
        training_type="supervised", lazy_loading=True,
        input_dim=2, output_dim=2
    )

    # batch size --> from 10 to 20
    for batch_size in range(10, 21):
        lazy_loader = DataLoader({"batch_size": batch_size}, dataset)

        for idx, batch in enumerate(lazy_loader):
            assert isinstance(batch, tuple)
            assert len(batch) == 2

            assert isinstance(batch[0], torch.Tensor)
            assert batch[0].dtype == torch.float32

            if batch_size*(idx+1) <= len(lazy_loader):
                assert batch[0].shape == torch.Size((batch_size, 1, 32, 32))
            else:
                batch_length = len(lazy_loader) % batch_size
                assert batch[0].shape == torch.Size((batch_length, 1, 32, 32))


def test_lazy_loader_3d():
    dataset = dataset_to_modify(
        training_type="supervised", lazy_loading=True,
        input_dim=3, output_dim=2
    )

    # batch size --> from 10 to 20
    for batch_size in range(10, 21):
        lazy_loader = DataLoader({"batch_size": batch_size}, dataset)

        for idx, batch in enumerate(lazy_loader):
            assert isinstance(batch, tuple)
            assert len(batch) == 2

            assert isinstance(batch[0], torch.Tensor)
            assert batch[0].dtype == torch.float32

            if batch_size*(idx+1) <= len(lazy_loader):
                batch_length = LENGHT_TENSOR_3D * batch_size
                assert batch[0].shape == torch.Size((batch_length, 1, 32, 32))
            else:
                batch_length = LENGHT_TENSOR_3D * (len(lazy_loader) % batch_size)
                assert batch[0].shape == torch.Size((batch_length, 1, 32, 32))


# -------------------- TENSOR LOADER -------------------- #

def test_tensor_loader_2d():
    dataset = dataset_to_modify(
        training_type="supervised", lazy_loading=False,
        input_dim=2, output_dim=2
    )

    # batch size --> from 10 to 20
    for batch_size in range(10, 21):
        tensor_loader = DataLoader({"batch_size": batch_size}, dataset)

        for idx, batch in enumerate(tensor_loader):
            assert isinstance(batch, tuple)
            assert len(batch) == 2

            assert isinstance(batch[0], torch.Tensor)
            assert batch[0].dtype == torch.float32

            if batch_size*(idx+1) <= len(tensor_loader):
                assert batch[0].shape == torch.Size((batch_size, 1, 32, 32))
            else:
                batch_length = len(tensor_loader) % batch_size
                assert batch[0].shape == torch.Size((batch_length, 1, 32, 32))


def test_tensor_loader_3d():
    dataset = dataset_to_modify(
        training_type="supervised", lazy_loading=False,
        input_dim=3, output_dim=2
    )

    # batch size --> from 10 to 20
    for batch_size in range(10, 21):
        tensor_loader = DataLoader({"batch_size": batch_size}, dataset)

        for idx, batch in enumerate(tensor_loader):
            assert isinstance(batch, tuple)
            assert len(batch) == 2

            assert isinstance(batch[0], torch.Tensor)
            assert batch[0].dtype == torch.float32

            if batch_size*(idx+1) <= len(tensor_loader):
                batch_length = LENGHT_TENSOR_3D * batch_size
                assert batch[0].shape == torch.Size((batch_length, 1, 32, 32))
            else:
                batch_length = LENGHT_TENSOR_3D * (len(tensor_loader) % batch_size)
                assert batch[0].shape == torch.Size((batch_length, 1, 32, 32))
