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

from src.loading.dataset.data_chopper import DataChopper, DataChopper2D, DataChopper25D, DataChopper3D
from src.loading.dataset.data_loader import TensorLoader
from src.loading.dataset import DataSet


# -------------------- CONSTANT -------------------- #

data_paths = {
    "tensor": {
        "2D": os.path.join(paths.DATA_TEST_PATH, "2D", "tensor", "input.pt"),
        "3D": os.path.join(paths.DATA_TEST_PATH, "3D", "tensor", "input.pt"),
    },
}


# -------------------- FIXTURES -------------------- #

@pytest.fixture(scope="function")
def input_target_tensor():
    input_tensor = TensorLoader()(data_paths["tensor"]["3D"])
    target_tensor = TensorLoader()(data_paths["tensor"]["3D"])

    dataset = DataSet(
        params={"file_type": "tensor", "input_dim": 3, "output_dim": 3},
        input_paths=[]
    )
    return dataset._adjust_shape(input_tensor), dataset._adjust_shape(target_tensor)


@pytest.fixture(scope="function")
def data_chopper():
    return DataChopper()


@pytest.fixture(scope="function")
def data_chopper_2d():
    return DataChopper2D()


@pytest.fixture(scope="function")
def data_chopper_25d():
    return DataChopper25D()


@pytest.fixture(scope="function")
def data_chopper_3d():
    return DataChopper3D()


# -------------------- DATA CHOPPER -------------------- #


def test_data_chopper(data_chopper, input_target_tensor):
    input_tensor, target_tensor = input_target_tensor

    with pytest.raises(NotImplementedError):
        data_chopper(input_tensor, target_tensor)


# -------------------- DATA CHOPPER 2D -------------------- #


def test_data_chopper_2d_chopping(data_chopper_2d, input_target_tensor):
    # input and target
    input_tensor, target_tensor = input_target_tensor
    input_tensor, target_tensor = data_chopper_2d._chopping(input_tensor, target_tensor)

    assert input_tensor.shape == torch.Size((32, 1, 32, 32))
    assert target_tensor.shape == torch.Size((32, 1, 32, 32))

    # input
    input_tensor, target_tensor = input_target_tensor
    with pytest.raises(ValueError):
        input_tensor, target_tensor = data_chopper_2d._chopping(input_tensor)

    # input
    input_tensor, target_tensor = input_target_tensor
    input_tensor = data_chopper_2d._chopping(input_tensor)

    assert input_tensor.shape == torch.Size((32, 1, 32, 32))


def test_data_chopper_2d(data_chopper_2d, input_target_tensor):
    # input and target
    input_tensor, target_tensor = input_target_tensor
    input_tensor, target_tensor = data_chopper_2d(input_tensor, target_tensor)

    assert input_tensor.shape == torch.Size((32, 1, 32, 32))
    assert target_tensor.shape == torch.Size((32, 1, 32, 32))

    # input
    input_tensor, target_tensor = input_target_tensor
    with pytest.raises(ValueError):
        input_tensor, target_tensor = data_chopper_2d(input_tensor)

    # input
    input_tensor, target_tensor = input_target_tensor
    input_tensor = data_chopper_2d(input_tensor)

    assert input_tensor.shape == torch.Size((32, 1, 32, 32))


# -------------------- DATA CHOPPER 2D -------------------- #

def test_data_chopper_25d_chopping(data_chopper_25d, input_target_tensor):
    # input and target
    input_tensor, target_tensor = input_target_tensor
    input_tensor, target_tensor = data_chopper_25d._chopping(input_tensor, target_tensor)

    assert isinstance(input_tensor, torch.Tensor)
    assert input_tensor.shape == torch.Size((28, 5, 32, 32))

    assert isinstance(target_tensor, torch.Tensor)
    assert target_tensor.shape == torch.Size((28, 1, 32, 32))

    # input
    input_tensor, target_tensor = input_target_tensor
    with pytest.raises(ValueError):
        input_tensor, target_tensor = data_chopper_25d._chopping(input_tensor)

    # input
    input_tensor, target_tensor = input_target_tensor
    input_tensor = data_chopper_25d._chopping(input_tensor)

    assert isinstance(input_tensor, torch.Tensor)
    assert input_tensor.shape == torch.Size((28, 5, 32, 32))


def test_data_chopper_25d(data_chopper_25d, input_target_tensor):
    # input and target
    input_tensor, target_tensor = input_target_tensor
    input_tensor, target_tensor = data_chopper_25d(input_tensor, target_tensor)

    assert isinstance(input_tensor, torch.Tensor)
    assert input_tensor.shape == torch.Size((28, 5, 32, 32))

    assert isinstance(target_tensor, torch.Tensor)
    assert target_tensor.shape == torch.Size((28, 1, 32, 32))

    # input
    input_tensor, target_tensor = input_target_tensor
    with pytest.raises(ValueError):
        input_tensor, target_tensor = data_chopper_25d(input_tensor)

    # input
    input_tensor, target_tensor = input_target_tensor
    input_tensor = data_chopper_25d(input_tensor)

    assert isinstance(input_tensor, torch.Tensor)
    assert input_tensor.shape == torch.Size((28, 5, 32, 32))

# -------------------- DATASET 3D -------------------- #


def test_data_chopper_3d_chopping(data_chopper_3d, input_target_tensor):
    pass


def test_data_chopper_3d(data_chopper_3d, input_target_tensor):
    pass
