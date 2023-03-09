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

from src.loading.dataset import DataSet, DataSetUnsupervised, DataSetSupervised
from src.loading.dataset.data_loader import TensorLoader


# -------------------- CONSTANT -------------------- #

data_paths = {
    "image": {
        "2D": os.path.join(paths.DATA_TEST_PATH, "2D", "image", "input.png"),
        "3D": os.path.join(paths.DATA_TEST_PATH, "3D", "image", "input.png"),
    },
    "tensor": {
        "2D": os.path.join(paths.DATA_TEST_PATH, "2D", "tensor", "input.pt"),
        "3D": os.path.join(paths.DATA_TEST_PATH, "3D", "tensor", "input.pt"),
    },
}


# -------------------- FIXTURES -------------------- #

@pytest.fixture(scope="function")
def dataset():
    return DataSet(
        params={"file_type": "tensor", "input_dim": 2, "output_dim": 2},
        input_paths=[data_paths["image"]["2D"]]
    )


@pytest.fixture(scope="function")
def dataset_unsupervised():
    return DataSetUnsupervised(
        params={"file_type": "tensor", "input_dim": 2, "output_dim": 2},
        input_paths=[data_paths["tensor"]["2D"], data_paths["image"]["2D"], data_paths["tensor"]["3D"]]
    )


@pytest.fixture(scope="function")
def dataset_supervised():
    return DataSetSupervised(
        params={"file_type": "tensor", "input_dim": 2, "output_dim": 2},
        input_paths=[data_paths["tensor"]["2D"], data_paths["image"]["2D"], data_paths["tensor"]["3D"]],
        target_paths=[data_paths["tensor"]["2D"], data_paths["image"]["2D"], data_paths["tensor"]["3D"]]
    )


# -------------------- DATASET -------------------- #


def test_dataset(dataset):
    assert len(dataset) == 1

    with pytest.raises(NotImplementedError):
        input_tensor, target_tensor = dataset[0]


def test_dataset_verify_shape_2d(dataset):
    dataset._params["input_dim"] = 2

    # 2D -> shape == 1
    tensor = torch.Tensor(32)
    with pytest.raises(ValueError):
        dataset._verify_shape(tensor)

    # 2D -> shape == 4
    tensor = torch.Tensor(3, 3, 32, 32)
    with pytest.raises(ValueError):
        dataset._verify_shape(tensor)

    # 2D -> shape == 5
    tensor = torch.Tensor(1, 1, 32, 32, 32)
    with pytest.raises(ValueError):
        dataset._verify_shape(tensor)

    # 2D -> shape with + 2 > 5
    tensor = torch.Tensor(20, 32, 32)
    with pytest.raises(ValueError):
        dataset._verify_shape(tensor)

    # 2D -> shape with + 2 > 5
    tensor = torch.Tensor(32, 32, 20)
    with pytest.raises(ValueError):
        dataset._verify_shape(tensor)

    # 2D -> shape with + 2 > 5
    tensor = torch.Tensor(1, 20, 32, 32)
    with pytest.raises(ValueError):
        dataset._verify_shape(tensor)

    # 3D instead of 2D
    tensor = TensorLoader()(data_paths["tensor"]["3D"])
    with pytest.raises(ValueError):
        dataset._verify_shape(tensor)


def test_dataset_adjust_shape_2d(dataset):
    dataset._params["input_dim"] = 2

    # 2D -> shape == 2
    tensor = torch.Tensor(32, 32)
    assert dataset._adjust_shape(tensor).shape == torch.Size((1, 1, 32, 32))

    # 2D -> shape == 3
    tensor = torch.Tensor(1, 32, 32)
    assert dataset._adjust_shape(tensor).shape == torch.Size((1, 1, 32, 32))

    # 2D -> shape == 3
    tensor = torch.Tensor(32, 32, 3)
    assert dataset._adjust_shape(tensor).shape == torch.Size((1, 3, 32, 32))

    # 2D -> shape == 3
    tensor = torch.Tensor(3, 32, 32)
    assert dataset._adjust_shape(tensor).shape == torch.Size((1, 3, 32, 32))

    # 2D -> shape == 4
    tensor = torch.Tensor(1, 3, 32, 32)
    assert dataset._adjust_shape(tensor).shape == torch.Size((1, 3, 32, 32))

    # 2D -> shape == 4
    tensor = torch.Tensor(1, 32, 32, 3)
    assert dataset._adjust_shape(tensor).shape == torch.Size((1, 3, 32, 32))


def test_dataset_verify_shape_3d(dataset):
    dataset._params["input_dim"] = 3

    # 2D -> shape == 1
    tensor = torch.Tensor(32)
    with pytest.raises(ValueError):
        dataset._verify_shape(tensor)

    # 2D -> shape == 2
    tensor = torch.Tensor(32, 32)
    with pytest.raises(ValueError):
        dataset._verify_shape(tensor)

    # 2D -> shape == 5
    tensor = torch.Tensor(3, 3, 32, 32, 32)
    with pytest.raises(ValueError):
        dataset._verify_shape(tensor)

    # 2D -> shape == 6
    tensor = torch.Tensor(1, 1, 32, 32, 32, 32)
    with pytest.raises(ValueError):
        dataset._verify_shape(tensor)

    # 2D -> shape with + 3 > 5
    tensor = torch.Tensor(20, 32, 32, 32)
    with pytest.raises(ValueError):
        dataset._verify_shape(tensor)

    # 2D -> shape with + 3 > 5
    tensor = torch.Tensor(32, 32, 32, 20)
    with pytest.raises(ValueError):
        dataset._verify_shape(tensor)

    # 2D -> shape with + 3 > 5
    tensor = torch.Tensor(1, 20, 32, 32, 32)
    with pytest.raises(ValueError):
        dataset._verify_shape(tensor)

    # 2D instead of 3D
    tensor = TensorLoader()(data_paths["tensor"]["2D"])
    with pytest.raises(ValueError):
        dataset._verify_shape(tensor)


def test_dataset_adjust_shape_3d(dataset):
    dataset._params["input_dim"] = 3

    # 2D -> shape == 3
    tensor = torch.Tensor(32, 32, 32)
    assert dataset._adjust_shape(tensor).shape == torch.Size((1, 1, 32, 32, 32))

    # 2D -> shape == 4
    tensor = torch.Tensor(1, 32, 32, 32)
    assert dataset._adjust_shape(tensor).shape == torch.Size((1, 1, 32, 32, 32))

    # 2D -> shape == 4
    tensor = torch.Tensor(32, 32, 32, 3)
    assert dataset._adjust_shape(tensor).shape == torch.Size((1, 3, 32, 32, 32))

    # 2D -> shape == 4
    tensor = torch.Tensor(3, 32, 32, 32)
    assert dataset._adjust_shape(tensor).shape == torch.Size((1, 3, 32, 32, 32))

    # 2D -> shape == 5
    tensor = torch.Tensor(1, 3, 32, 32, 32)
    assert dataset._adjust_shape(tensor).shape == torch.Size((1, 3, 32, 32, 32))

    # 2D -> shape == 5
    tensor = torch.Tensor(1, 32, 32, 32, 3)
    assert dataset._adjust_shape(tensor).shape == torch.Size((1, 3, 32, 32, 32))


# -------------------- DATASET UNSUPERVISED -------------------- #

def test_dataset_unsupervised_getitem(dataset_unsupervised):
    with pytest.raises(ValueError):
        input_tensor, target_tensor = dataset_unsupervised[0]
    input_tensor = dataset_unsupervised[0]

    assert isinstance(input_tensor, torch.Tensor)
    assert input_tensor.dtype == torch.float32


# -------------------- DATASET SUPERVISED -------------------- #

def test_dataset_supervised_getitem(dataset_supervised):
    input_tensor, target_tensor = dataset_supervised[0]

    assert isinstance(input_tensor, torch.Tensor)
    assert input_tensor.dtype == torch.float32

    assert isinstance(target_tensor, torch.Tensor)
    assert target_tensor.dtype == torch.float32


# -------------------- DATASET 2D -------------------- #

def test_dataset_getitem_2d():
    dataset = DataSetSupervised(
        params={"file_type": "tensor", "input_dim": 2, "output_dim": 2},
        input_paths=[data_paths["tensor"]["2D"], data_paths["image"]["2D"], data_paths["tensor"]["3D"]],
        target_paths=[data_paths["tensor"]["2D"], data_paths["image"]["2D"], data_paths["tensor"]["3D"]]
    )

    # 2D tensor
    input_tensor, target_tensor = dataset[0]

    assert isinstance(input_tensor, torch.Tensor)
    assert input_tensor.shape == torch.Size((1, 1, 32, 32))

    assert isinstance(target_tensor, torch.Tensor)
    assert target_tensor.shape == torch.Size((1, 1, 32, 32))

    # Not a .pt file
    with pytest.raises(ValueError):
        target_tensor = dataset[1]

    # 3D instead of 2D
    with pytest.raises(ValueError):
        target_tensor = dataset[2]


# -------------------- DATASET 3D -------------------- #

def test_dataset_getitem_3d():
    dataset = DataSetSupervised(
        params={"file_type": "tensor", "input_dim": 3, "output_dim": 2},
        input_paths=[data_paths["tensor"]["2D"], data_paths["image"]["2D"], data_paths["tensor"]["3D"]],
        target_paths=[data_paths["tensor"]["2D"], data_paths["image"]["2D"], data_paths["tensor"]["3D"]]
    )

    # 3D tensor
    input_tensor, target_tensor = dataset[2]

    # Not a .pt file
    with pytest.raises(ValueError):
        input_tensor, target_tensor = dataset[0]

    # 3D instead of 2D
    with pytest.raises(ValueError):
        input_tensor, target_tensor = dataset[1]


def test_dataset_getitem_3d_2d():
    dataset = DataSetSupervised(
        params={"file_type": "tensor", "input_dim": 3, "output_dim": 2},
        input_paths=[data_paths["tensor"]["3D"]],
        target_paths=[data_paths["tensor"]["3D"]]
    )

    # 3D tensor
    input_tensor, target_tensor = dataset[0]

    assert isinstance(input_tensor, torch.Tensor)
    assert input_tensor.shape == torch.Size((32, 1, 32, 32))

    assert isinstance(target_tensor, torch.Tensor)
    assert target_tensor.shape == torch.Size((32, 1, 32, 32))


def test_dataset_getitem_3d_25d():
    dataset = DataSetSupervised(
        params={"file_type": "tensor", "input_dim": 3, "output_dim": 2.5},
        input_paths=[data_paths["tensor"]["3D"]],
        target_paths=[data_paths["tensor"]["3D"]]
    )

    # 3D tensor
    input_tensor, target_tensor = dataset[0]

    assert isinstance(input_tensor, torch.Tensor)
    assert input_tensor.shape == torch.Size((28, 5, 32, 32))

    assert isinstance(target_tensor, torch.Tensor)
    assert target_tensor.shape == torch.Size((28, 1, 32, 32))


def test_dataset_3d_3d_getitem():
    pass
