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

from src.loading.dataset.data_chopper import DataChopper, \
    DataChopper2D, DataChopper25D, DataChopper3D

from src.loading.dataset import SupervisedDataSet

# -------------------- CONSTANT -------------------- #

DATA_PATHS = {
    "tensor": {
        "2D": os.path.join(paths.TEST_DATA_PATH, "data_dim", "2D", "tensor", "input.pt"),
        "3D": os.path.join(paths.TEST_DATA_PATH, "data_dim", "3D", "tensor", "input.pt"),
    },
}


# -------------------- FIXTURES -------------------- #

@pytest.fixture(scope="function")
def input_target_tensors():
    dataset = SupervisedDataSet(
        params={
            "training_type": "supervised", "file_type": "tensor", "lazy_loading": True,
            "input_dim": 3, "output_dim": 2, "out_channels": 1
        },
        inputs=[DATA_PATHS["tensor"]["3D"]],
        targets=[DATA_PATHS["tensor"]["3D"]]
    )
    return dataset._get_data(dataset._inputs[0]), dataset._get_data(dataset._targets[0])


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


def test_data_chopper(data_chopper, input_target_tensors):
    input_tensor, target_tensor = input_target_tensors

    with pytest.raises(NotImplementedError):
        data_chopper(input_tensor, target_tensor)


# -------------------- DATA CHOPPER 2D -------------------- #


def test_data_chopper_2d_chop(data_chopper_2d, input_target_tensors):
    # input and target
    input_tensor, target_tensor = input_target_tensors
    input_tensor, target_tensor = data_chopper_2d._chop(input_tensor, target_tensor)

    assert input_tensor.shape == torch.Size((32, 1, 32, 32))
    assert target_tensor.shape == torch.Size((32, 1, 32, 32))

    # input
    input_tensor, target_tensor = input_target_tensors
    with pytest.raises(ValueError):
        input_tensor, target_tensor = data_chopper_2d._chop(input_tensor)

    # input
    input_tensor, target_tensor = input_target_tensors
    input_tensor = data_chopper_2d._chop(input_tensor)

    assert input_tensor.shape == torch.Size((32, 1, 32, 32))


def test_data_chopper_2d(data_chopper_2d, input_target_tensors):
    # input and target
    input_tensor, target_tensor = input_target_tensors
    input_tensor, target_tensor = data_chopper_2d(input_tensor, target_tensor)

    assert input_tensor.shape == torch.Size((32, 1, 32, 32))
    assert target_tensor.shape == torch.Size((32, 1, 32, 32))

    # input
    input_tensor, target_tensor = input_target_tensors
    with pytest.raises(ValueError):
        input_tensor, target_tensor = data_chopper_2d(input_tensor)

    # input
    input_tensor, target_tensor = input_target_tensors
    input_tensor = data_chopper_2d(input_tensor)

    assert input_tensor.shape == torch.Size((32, 1, 32, 32))


# -------------------- DATA CHOPPER 2D -------------------- #

def test_data_chopper_25d_chop(data_chopper_25d, input_target_tensors):
    # input and target
    input_tensor, target_tensor = input_target_tensors
    input_tensor, target_tensor = data_chopper_25d._chop(input_tensor, target_tensor)

    assert isinstance(input_tensor, torch.Tensor)
    assert input_tensor.shape == torch.Size((28, 5, 32, 32))

    assert isinstance(target_tensor, torch.Tensor)
    assert target_tensor.shape == torch.Size((28, 1, 32, 32))

    # input
    input_tensor, target_tensor = input_target_tensors
    with pytest.raises(ValueError):
        input_tensor, target_tensor = data_chopper_25d._chop(input_tensor)

    # input
    input_tensor, target_tensor = input_target_tensors
    input_tensor = data_chopper_25d._chop(input_tensor)

    assert isinstance(input_tensor, torch.Tensor)
    assert input_tensor.shape == torch.Size((28, 5, 32, 32))


def test_data_chopper_25d(data_chopper_25d, input_target_tensors):
    # input and target
    input_tensor, target_tensor = input_target_tensors
    input_tensor, target_tensor = data_chopper_25d(input_tensor, target_tensor)

    assert isinstance(input_tensor, torch.Tensor)
    assert input_tensor.shape == torch.Size((28, 5, 32, 32))

    assert isinstance(target_tensor, torch.Tensor)
    assert target_tensor.shape == torch.Size((28, 1, 32, 32))

    # input
    input_tensor, target_tensor = input_target_tensors
    with pytest.raises(ValueError):
        input_tensor, target_tensor = data_chopper_25d(input_tensor)

    # input
    input_tensor, target_tensor = input_target_tensors
    input_tensor = data_chopper_25d(input_tensor)

    assert isinstance(input_tensor, torch.Tensor)
    assert input_tensor.shape == torch.Size((28, 5, 32, 32))

# -------------------- DATASET 3D -------------------- #


def test_data_chopper_3d_chop(data_chopper_3d, input_target_tensors):
    pass


def test_data_chopper_3d(data_chopper_3d, input_target_tensors):
    pass
