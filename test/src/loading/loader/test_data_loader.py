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

from src.loading.loader.data_loader import LazyLoader, TensorLoader
from src.loading.dataset import DataSetSupervised


# -------------------- CONSTANT -------------------- #

data_paths = {
    "2D": os.path.join(paths.DATA_TEST_PATH, "2D", "tensor", "input.pt"),
    "3D": os.path.join(paths.DATA_TEST_PATH, "3D", "tensor", "input.pt"),
}

LENGHT_TENSOR_3D = 32


# -------------------- FIXTURES -------------------- #

@pytest.fixture(scope="function")
def dataset2d():
    return DataSetSupervised(
        params={"file_type": "tensor", "input_dim": 2, "output_dim": 2},
        input_paths=[data_paths["2D"] for i in range(100)],
        target_paths=[data_paths["2D"] for i in range(100)]
    )


@pytest.fixture(scope="function")
def dataset3d2d():
    return DataSetSupervised(
        params={"file_type": "tensor", "input_dim": 3, "output_dim": 2},
        input_paths=[data_paths["3D"] for i in range(100)],
        target_paths=[data_paths["3D"] for i in range(100)]
    )


@pytest.fixture(scope="function")
def dataset3d25d():
    return DataSetSupervised(
        params={"file_type": "tensor", "input_dim": 3, "output_dim": 2.5},
        input_paths=[data_paths["3D"] for i in range(100)],
        target_paths=[data_paths["3D"] for i in range(100)]
    )


@pytest.fixture(scope="function")
def dataset3d3d():
    return DataSetSupervised(
        params={"file_type": "tensor", "input_dim": 3, "output_dim": 3},
        input_paths=[data_paths["3D"] for i in range(100)],
        target_paths=[data_paths["3D"] for i in range(100)]
    )


# -------------------- LAZY LOADER -------------------- #

def test_lazy_loader_2d(dataset2d):
    # batch size --> from 10 to 20
    for batch_size in range(10, 21):
        lazy_loader = LazyLoader({"batch_size": batch_size}, dataset2d)

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


def test_lazy_loader_3d(dataset3d2d):
    # batch size --> from 10 to 20
    for batch_size in range(10, 21):
        lazy_loader = LazyLoader({"batch_size": batch_size}, dataset3d2d)

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

def test_tensor_loader_2d(dataset2d):
    # batch size --> from 10 to 20
    for batch_size in range(10, 21):
        tensor_loader = TensorLoader({"batch_size": batch_size, "workers": 2}, dataset2d)

        for idx, batch in enumerate(tensor_loader):
            print(type(batch))
            assert isinstance(batch, torch.Tensor)
            assert len(batch) == 2

            assert isinstance(batch[0], torch.Tensor)
            assert batch[0].dtype == torch.float32

            if batch_size*(idx+1) <= len(tensor_loader):
                assert batch[0].shape == torch.Size((batch_size, 1, 32, 32))
            else:
                batch_length = len(tensor_loader) % batch_size
                assert batch[0].shape == torch.Size((batch_length, 1, 32, 32))


def test_tensor_loader_3d(dataset3d2d):
    # batch size --> from 10 to 20
    for batch_size in range(10, 21):
        tensor_loader = TensorLoader({"batch_size": batch_size, "workers": 2}, dataset3d2d)

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
