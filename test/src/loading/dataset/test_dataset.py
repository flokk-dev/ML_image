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

from src.loading.dataset.dataset import DataSet
from src.loading.dataset import DataSet2D, DataSet3D

from src.loading.dataset.data_loader import TensorLoader


# -------------------- CONSTANT -------------------- #

DATA_PATHS = {
    "image": {
        "2D": os.path.join(paths.TEST_PATH, "data_dim", "2D", "image", "input.png")
    },
    "tensor": {
        "2D": os.path.join(paths.TEST_PATH, "data_dim", "2D", "tensor", "input.pt"),
        "3D": os.path.join(paths.TEST_PATH, "data_dim", "3D", "tensor", "input.pt"),
    },
}


# -------------------- FIXTURES -------------------- #

@pytest.fixture(scope="function")
def dataset():
    return DataSet(
        params={
            "training_type": "supervised", "file_type": "tensor", "lazy_loading": True,
            "input_dim": 2, "output_dim": 2
        },
        inputs=[DATA_PATHS["tensor"]["2D"] for i in range(10)],
        targets=[DATA_PATHS["tensor"]["2D"] for i in range(10)]
    )


@pytest.fixture(scope="function")
def dataset_2d():
    return DataSet2D(
        params={
            "training_type": "supervised", "file_type": "tensor", "lazy_loading": True,
            "input_dim": 2, "output_dim": 2
        },
        inputs=[DATA_PATHS["tensor"]["2D"], DATA_PATHS["image"]["2D"], DATA_PATHS["tensor"]["3D"]],
        targets=[DATA_PATHS["tensor"]["2D"], DATA_PATHS["image"]["2D"], DATA_PATHS["tensor"]["3D"]]
    )


@pytest.fixture(scope="function")
def dataset_3d():
    return DataSet3D(
        params={
            "training_type": "supervised", "file_type": "tensor", "lazy_loading": True,
            "input_dim": 3, "output_dim": 2
        },
        inputs=[DATA_PATHS["tensor"]["3D"], DATA_PATHS["image"]["2D"], DATA_PATHS["tensor"]["2D"]],
        targets=[DATA_PATHS["tensor"]["3D"], DATA_PATHS["image"]["2D"], DATA_PATHS["tensor"]["2D"]]
    )


def dataset_to_modify(training_type, file_type, lazy_loading, input_dim, output_dim):
    datasets = {2: DataSet2D, 3: DataSet3D}
    return datasets[input_dim](
        params={
            "training_type": training_type, "file_type": file_type, "lazy_loading": lazy_loading,
            "input_dim": input_dim, "output_dim": output_dim
        },
        inputs=[DATA_PATHS[file_type][f"{str(input_dim)}D"] for i in range(10)],
        targets=[DATA_PATHS[file_type][f"{str(input_dim)}D"] for i in range(10)]
    )


# -------------------- DATASET -------------------- #

def test_dataset(dataset):
    with pytest.raises(NotImplementedError):
        input_tensor, target_tensor = dataset.__getitem__(0)

    with pytest.raises(NotImplementedError):
        input_tensor, target_tensor = dataset[0]


# -------------------- DATASET 2D -------------------- #

def test_dataset_verify_shape_2d(dataset_2d):
    # 2D -> shape == 1
    tensor = torch.Tensor(32)
    with pytest.raises(ValueError):
        dataset_2d._verify_shape(tensor)

    # 2D -> shape == 4
    tensor = torch.Tensor(3, 3, 32, 32)
    with pytest.raises(ValueError):
        dataset_2d._verify_shape(tensor)

    # 2D -> shape == 5
    tensor = torch.Tensor(1, 1, 32, 32, 32)
    with pytest.raises(ValueError):
        dataset_2d._verify_shape(tensor)

    # 2D -> shape with + 2 > 5
    tensor = torch.Tensor(20, 32, 32)
    with pytest.raises(ValueError):
        dataset_2d._verify_shape(tensor)

    # 2D -> shape with + 2 > 5
    tensor = torch.Tensor(32, 32, 20)
    with pytest.raises(ValueError):
        dataset_2d._verify_shape(tensor)

    # 2D -> shape with + 2 > 5
    tensor = torch.Tensor(1, 20, 32, 32)
    with pytest.raises(ValueError):
        dataset_2d._verify_shape(tensor)

    # 3D instead of 2D
    tensor = TensorLoader()(DATA_PATHS["tensor"]["3D"])
    with pytest.raises(ValueError):
        dataset_2d._verify_shape(tensor)


def test_dataset_adjust_shape_2d(dataset_2d):
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


def test_dataset_get_data_2d(dataset_2d):
    # 2D tensor
    input_tensor = dataset_2d._get_data(dataset_2d._inputs[0])

    assert isinstance(input_tensor, torch.Tensor)
    assert input_tensor.shape == torch.Size((1, 1, 32, 32))

    # Not a .pt file
    with pytest.raises(ValueError):
        tensor = dataset_2d._get_data(dataset_2d._inputs[1])

    # 3D instead of 2D
    with pytest.raises(ValueError):
        tensor = dataset_2d._get_data(dataset_2d._inputs[2])


def test_dataset_getitem_2d(dataset_2d):
    # 2D tensor
    input_tensor, target_tensor = dataset_2d[0]

    assert isinstance(input_tensor, torch.Tensor)
    assert input_tensor.shape == torch.Size((1, 1, 32, 32))

    assert isinstance(target_tensor, torch.Tensor)
    assert target_tensor.shape == torch.Size((1, 1, 32, 32))


# -------------------- DATASET 3D -------------------- #

def test_dataset_verify_shape_3d(dataset_3d):
    # 2D -> shape == 1
    tensor = torch.Tensor(32)
    with pytest.raises(ValueError):
        dataset_3d._verify_shape(tensor)

    # 2D -> shape == 2
    tensor = torch.Tensor(32, 32)
    with pytest.raises(ValueError):
        dataset_3d._verify_shape(tensor)

    # 2D -> shape == 5
    tensor = torch.Tensor(3, 3, 32, 32, 32)
    with pytest.raises(ValueError):
        dataset_3d._verify_shape(tensor)

    # 2D -> shape == 6
    tensor = torch.Tensor(1, 1, 32, 32, 32, 32)
    with pytest.raises(ValueError):
        dataset_3d._verify_shape(tensor)

    # 2D -> shape with + 3 > 5
    tensor = torch.Tensor(20, 32, 32, 32)
    with pytest.raises(ValueError):
        dataset_3d._verify_shape(tensor)

    # 2D -> shape with + 3 > 5
    tensor = torch.Tensor(32, 32, 32, 20)
    with pytest.raises(ValueError):
        dataset_3d._verify_shape(tensor)

    # 2D -> shape with + 3 > 5
    tensor = torch.Tensor(1, 20, 32, 32, 32)
    with pytest.raises(ValueError):
        dataset_3d._verify_shape(tensor)

    # 2D instead of 3D
    tensor = TensorLoader()(DATA_PATHS["tensor"]["2D"])
    with pytest.raises(ValueError):
        dataset_3d._verify_shape(tensor)


def test_dataset_adjust_shape_3d(dataset_3d):
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


def test_dataset_load_data_3d(dataset_3d):
    # 3D tensor
    input_tensor = dataset_3d._get_data(dataset_3d._inputs[0])

    assert isinstance(input_tensor, torch.Tensor)
    assert input_tensor.shape == torch.Size((1, 1, 32, 32, 32))

    # Not a .pt file
    with pytest.raises(ValueError):
        tensor = dataset_3d._get_data(dataset_3d._inputs[1])

    # 2D instead of 3D
    with pytest.raises(ValueError):
        tensor = dataset_3d._get_data(dataset_3d._inputs[2])


def test_dataset_getitem_3d_2d():
    dataset = dataset_to_modify(
        training_type="supervised", file_type="tensor", lazy_loading=True,
        input_dim=3, output_dim=2
    )

    # 3D tensor
    input_tensor, target_tensor = dataset[0]

    assert isinstance(input_tensor, torch.Tensor)
    assert input_tensor.shape == torch.Size((32, 1, 32, 32))

    assert isinstance(target_tensor, torch.Tensor)
    assert target_tensor.shape == torch.Size((32, 1, 32, 32))


def test_dataset_getitem_3d_25d():
    dataset = dataset_to_modify(
        training_type="supervised", file_type="tensor", lazy_loading=True,
        input_dim=3, output_dim=2.5
    )

    # 3D tensor
    input_tensor, target_tensor = dataset[0]

    assert isinstance(input_tensor, torch.Tensor)
    assert input_tensor.shape == torch.Size((28, 5, 32, 32))

    assert isinstance(target_tensor, torch.Tensor)
    assert target_tensor.shape == torch.Size((28, 1, 32, 32))


def test_dataset_3d_3d_getitem():
    pass


# -------------------- LAZY DATASET -------------------- #

def test_lazy_dataset():
    dataset = dataset_to_modify(
        training_type="supervised", file_type="tensor", lazy_loading=True,
        input_dim=2, output_dim=2
    )

    for idx in range(len(dataset._inputs)):
        assert isinstance(dataset._inputs[idx], str)
        assert isinstance(dataset._targets[idx], str)


# -------------------- TENSOR DATASET -------------------- #

def test_tensor_dataset():
    dataset = dataset_to_modify(
        training_type="supervised", file_type="tensor", lazy_loading=False,
        input_dim=2, output_dim=2
    )

    for idx in range(len(dataset._inputs)):
        assert isinstance(dataset._inputs[idx], torch.Tensor)
        assert isinstance(dataset._targets[idx], torch.Tensor)


# -------------------- UNSUPERVISED DATASET -------------------- #

def test_unsupervised_dataset():
    dataset = dataset_to_modify(
        training_type="unsupervised", file_type="tensor", lazy_loading=True,
        input_dim=2, output_dim=2
    )

    # Input
    input_tensor = dataset[0]

    assert isinstance(input_tensor, torch.Tensor)
    assert input_tensor.shape == torch.Size((1, 1, 32, 32))

    # Input and target
    with pytest.raises(ValueError):
        input_tensor, target_tensor = dataset[0]


# -------------------- SUPERVISED DATASET -------------------- #

def test_supervised_dataset():
    dataset = dataset_to_modify(
        training_type="supervised", file_type="tensor", lazy_loading=True,
        input_dim=2, output_dim=2
    )

    # Input and target
    input_tensor, target_tensor = dataset[0]

    assert isinstance(input_tensor, torch.Tensor)
    assert input_tensor.shape == torch.Size((1, 1, 32, 32))

    assert isinstance(target_tensor, torch.Tensor)
    assert target_tensor.shape == torch.Size((1, 1, 32, 32))
