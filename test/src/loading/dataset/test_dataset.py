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

from src.loading.dataset.data_loader import TensorLoader
from src.loading.dataset import DataSet, DataSet2D, DataSet3D


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
        params={"file_type": "tensor", "dim": 2},
        input_paths=[data_paths["image"]["2D"], data_paths["tensor"]["2D"]],
        target_paths=[data_paths["image"]["2D"], data_paths["tensor"]["2D"]],
    )


@pytest.fixture(scope="function")
def dataset_2d():
    return DataSet2D(
        params={"file_type": "tensor", "dim": 2},
        input_paths=[data_paths["tensor"]["2D"], data_paths["image"]["2D"], data_paths["tensor"]["3D"]],
        target_paths=[data_paths["tensor"]["2D"], data_paths["image"]["2D"], data_paths["tensor"]["3D"]],
    )


@pytest.fixture(scope="function")
def dataset_3d():
    return DataSet3D(
        params={"file_type": "tensor", "dim": 2},
        input_paths=[data_paths["image"]["2D"], data_paths["tensor"]["2D"]],
        target_paths=[data_paths["image"]["2D"], data_paths["tensor"]["2D"]],
    )


# -------------------- DATASET -------------------- #


def test_dataset(dataset):
    assert len(dataset) == 2

    with pytest.raises(NotImplementedError):
        input_tensor, target_tensor = dataset[0]


# -------------------- DATASET 2D -------------------- #


def test_dataset_2d_adjust_shape(dataset_2d):
    # 2D -> shape == 1
    tensor = torch.Tensor(32)
    with pytest.raises(ValueError):
        dataset_2d._adjust_shape(tensor)

    # 2D -> shape == 2
    tensor = torch.Tensor(32, 32)
    assert dataset_2d._adjust_shape(tensor).shape == torch.Size((1, 1, 32, 32))

    # 2D -> shape == 3
    tensor = torch.Tensor(1, 32, 32)
    assert dataset_2d._adjust_shape(tensor).shape == torch.Size((1, 1, 32, 32))

    # 2D -> shape == 3
    tensor = torch.Tensor(32, 32, 3)
    assert dataset_2d._adjust_shape(tensor).shape == torch.Size((1, 3, 32, 32))

    # 2D -> shape == 3
    tensor = torch.Tensor(3, 32, 32)
    assert dataset_2d._adjust_shape(tensor).shape == torch.Size((1, 3, 32, 32))

    # 2D -> shape == 4
    tensor = torch.Tensor(1, 3, 32, 32)
    assert dataset_2d._adjust_shape(tensor).shape == torch.Size((1, 3, 32, 32))

    # 2D -> shape == 4
    tensor = torch.Tensor(1, 32, 32, 3)
    assert dataset_2d._adjust_shape(tensor).shape == torch.Size((1, 3, 32, 32))

    # 2D -> shape == 4
    tensor = torch.Tensor(3, 3, 32, 32)
    with pytest.raises(ValueError):
        dataset_2d._adjust_shape(tensor)

    # 2D -> shape == 5
    tensor = torch.Tensor(1, 1, 32, 32, 32)
    with pytest.raises(ValueError):
        dataset_2d._adjust_shape(tensor)

    # 2D -> shape with + 2 > 5
    tensor = torch.Tensor(20, 32, 32)
    with pytest.raises(ValueError):
        dataset_2d._adjust_shape(tensor)

    # 2D -> shape with + 2 > 5
    tensor = torch.Tensor(32, 32, 20)
    with pytest.raises(ValueError):
        dataset_2d._adjust_shape(tensor)

    # 2D -> shape with + 2 > 5
    tensor = torch.Tensor(1, 20, 32, 32)
    with pytest.raises(ValueError):
        dataset_2d._adjust_shape(tensor)

    # 3D instead of 2D
    tensor = TensorLoader()(data_paths["tensor"]["3D"])
    with pytest.raises(ValueError):
        dataset_2d._adjust_shape(tensor)


def test_dataset_2d_getitem(dataset_2d):
    # 2D
    input_tensor, target_tensor = dataset_2d[0]

    assert isinstance(input_tensor, torch.Tensor)
    assert isinstance(target_tensor, torch.Tensor)

    assert input_tensor.shape == torch.Size((1, 1, 32, 32))
    assert input_tensor.shape == torch.Size((1, 1, 32, 32))

    # Not a tensor
    with pytest.raises(ValueError):
        input_tensor, target_tensor = dataset_2d[1]

    # 3D instead of 2D
    with pytest.raises(ValueError):
        input_tensor, target_tensor = dataset_2d[2]


# -------------------- DATASET 3D -------------------- #


def test_dataset_3d_adjust_shape(dataset_3d):
    # 2D -> shape == 1
    tensor = torch.Tensor(32)
    with pytest.raises(ValueError):
        dataset_3d._adjust_shape(tensor)

    # 2D -> shape == 2
    tensor = torch.Tensor(32, 32)
    with pytest.raises(ValueError):
        dataset_3d._adjust_shape(tensor)

    # 2D -> shape == 3
    tensor = torch.Tensor(32, 32, 32)
    assert dataset_3d._adjust_shape(tensor).shape == torch.Size((1, 1, 32, 32, 32))

    # 2D -> shape == 4
    tensor = torch.Tensor(1, 32, 32, 32)
    assert dataset_3d._adjust_shape(tensor).shape == torch.Size((1, 1, 32, 32, 32))

    # 2D -> shape == 4
    tensor = torch.Tensor(32, 32, 32, 3)
    assert dataset_3d._adjust_shape(tensor).shape == torch.Size((1, 3, 32, 32, 32))

    # 2D -> shape == 4
    tensor = torch.Tensor(3, 32, 32, 32)
    assert dataset_3d._adjust_shape(tensor).shape == torch.Size((1, 3, 32, 32, 32))

    # 2D -> shape == 5
    tensor = torch.Tensor(1, 3, 32, 32, 32)
    assert dataset_3d._adjust_shape(tensor).shape == torch.Size((1, 3, 32, 32, 32))

    # 2D -> shape == 5
    tensor = torch.Tensor(1, 32, 32, 32, 3)
    assert dataset_3d._adjust_shape(tensor).shape == torch.Size((1, 3, 32, 32, 32))

    # 2D -> shape == 5
    tensor = torch.Tensor(3, 3, 32, 32, 32)
    with pytest.raises(ValueError):
        dataset_3d._adjust_shape(tensor)

    # 2D -> shape == 6
    tensor = torch.Tensor(1, 1, 32, 32, 32, 32)
    with pytest.raises(ValueError):
        dataset_3d._adjust_shape(tensor)

    # 2D -> shape with + 3 > 5
    tensor = torch.Tensor(20, 32, 32, 32)
    with pytest.raises(ValueError):
        dataset_3d._adjust_shape(tensor)

    # 2D -> shape with + 3 > 5
    tensor = torch.Tensor(32, 32, 32, 20)
    with pytest.raises(ValueError):
        dataset_3d._adjust_shape(tensor)

    # 2D -> shape with + 3 > 5
    tensor = torch.Tensor(1, 20, 32, 32, 32)
    with pytest.raises(ValueError):
        dataset_3d._adjust_shape(tensor)

    # 2D instead of 3D
    tensor = TensorLoader()(data_paths["tensor"]["2D"])
    with pytest.raises(ValueError):
        dataset_3d._adjust_shape(tensor)


def test_dataset_3d_getitem(dataset_3d):
    # Not a tensor
    with pytest.raises(ValueError):
        input_tensor, target_tensor = dataset_3d[0]

    # 2D instead of 3D
    with pytest.raises(ValueError):
        input_tensor, target_tensor = dataset_3d[1]


def test_dataset_3d_2d_getitem():
    # 3D to 2D
    dataset = DataSet3D(
        params={"file_type": "tensor", "dim": 2},
        input_paths=[data_paths["tensor"]["3D"]],
        target_paths=[data_paths["tensor"]["3D"]]
    )
    input_tensor, target_tensor = dataset[0]

    assert isinstance(input_tensor, torch.Tensor)
    assert input_tensor.shape == torch.Size((32, 1, 32, 32))

    assert isinstance(target_tensor, torch.Tensor)
    assert target_tensor.shape == torch.Size((32, 1, 32, 32))

    # 3D to 2D without target
    dataset = DataSet3D(
        params={"file_type": "tensor", "dim": 2},
        input_paths=[data_paths["tensor"]["3D"]]
    )

    with pytest.raises(ValueError):
        input_tensor, target_tensor = dataset[0]

    input_tensor = dataset[0]
    assert isinstance(input_tensor, torch.Tensor)
    assert input_tensor.shape == torch.Size((32, 1, 32, 32))


def test_dataset_3d_25d_getitem():
    # 3D to 2D
    dataset = DataSet3D(
        params={"file_type": "tensor", "dim": 2.5},
        input_paths=[data_paths["tensor"]["3D"]],
        target_paths=[data_paths["tensor"]["3D"]]
    )
    input_tensor, target_tensor = dataset[0]

    assert isinstance(input_tensor, torch.Tensor)
    assert input_tensor.shape == torch.Size((28, 5, 32, 32))

    assert isinstance(target_tensor, torch.Tensor)
    assert target_tensor.shape == torch.Size((28, 1, 32, 32))

    # 3D to 2D without target
    dataset = DataSet3D(
        params={"file_type": "tensor", "dim": 2.5},
        input_paths=[data_paths["tensor"]["3D"]]
    )

    with pytest.raises(ValueError):
        input_tensor, target_tensor = dataset[0]

    input_tensor = dataset[0]
    assert isinstance(input_tensor, torch.Tensor)
    assert input_tensor.shape == torch.Size((28, 5, 32, 32))


def test_dataset_3d_3d_getitem():
    pass
