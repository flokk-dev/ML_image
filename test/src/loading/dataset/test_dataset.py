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

from src.loading.dataset import DataSet, \
    UnsupervisedDataSet, SupervisedDataSet

from src.loading.dataset.file_loader import PtLoader


# -------------------- CONSTANT -------------------- #

DATA_PATHS = {
    "image": {
        "2D": os.path.join(paths.TEST_DATA_PATH, "data_dim", "2D", "image", "input.png")
    },
    "tensor": {
        "2D": os.path.join(paths.TEST_DATA_PATH, "data_dim", "2D", "tensor", "input.pt"),
        "3D": os.path.join(paths.TEST_DATA_PATH, "data_dim", "3D", "tensor", "input.pt"),
    },
}


# -------------------- FIXTURES -------------------- #

@pytest.fixture(scope="function")
def dataset():
    return DataSet(
        params={
            "file_type": "tensor", "lazy_loading": True,
            "input_dim": 2, "output_dim": 2
        },
        input_paths=[DATA_PATHS["tensor"]["2D"], DATA_PATHS["image"]["2D"], DATA_PATHS["tensor"]["3D"]],
        target_paths=[DATA_PATHS["tensor"]["2D"], DATA_PATHS["image"]["2D"], DATA_PATHS["tensor"]["3D"]]
    )


def dataset_to_modify(training_type, file_type, lazy_loading, input_dim, output_dim):
    datasets = {"unsupervised": UnsupervisedDataSet, "supervised": SupervisedDataSet}
    return datasets[training_type](
        params={
            "file_type": file_type, "lazy_loading": lazy_loading,
            "input_dim": input_dim, "output_dim": output_dim
        },
        input_paths=[DATA_PATHS[file_type][f"{str(input_dim)}D"] for i in range(10)],
        target_paths=[DATA_PATHS[file_type][f"{str(input_dim)}D"] for i in range(10)]
    )


# -------------------- DATASET -------------------- #

def test_dataset(dataset):
    with pytest.raises(NotImplementedError):
        input_tensor, target_tensor = dataset._collect_data_info()

    with pytest.raises(NotImplementedError):
        input_tensor, target_tensor = dataset.__getitem__(0)

    with pytest.raises(NotImplementedError):
        input_tensor, target_tensor = dataset[0]

    # Not a .pt file
    with pytest.raises(ValueError):
        tensor = dataset._get_data(dataset._inputs[1])


def test_dataset_verify_shape_wrong(dataset):
    # 2D -> shape == 1
    tensor = torch.Tensor(32)
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

    # 2D -> shape with + 2 > 1
    tensor = torch.Tensor(3, 3, 32, 32)
    with pytest.raises(ValueError):
        dataset._verify_shape(tensor)

    # 3D instead of 2D
    tensor = PtLoader()(DATA_PATHS["tensor"]["3D"])
    with pytest.raises(ValueError):
        dataset._verify_shape(tensor)


def test_dataset_verify_shape_wright(dataset):
    # 2D -> shape == 2
    tensor = torch.Tensor(32, 32)
    dataset._verify_shape(tensor)

    # 2D -> shape == 3
    tensor = torch.Tensor(1, 32, 32)
    dataset._verify_shape(tensor)

    # 2D -> shape == 3
    tensor = torch.Tensor(32, 32, 3)
    dataset._verify_shape(tensor)

    # 2D -> shape == 3
    tensor = torch.Tensor(3, 32, 32)
    dataset._verify_shape(tensor)

    # 2D -> shape == 4
    tensor = torch.Tensor(1, 3, 32, 32)
    dataset._verify_shape(tensor)

    # 2D -> shape == 4
    tensor = torch.Tensor(1, 32, 32, 3)
    dataset._verify_shape(tensor)


def test_dataset_adjust_shape(dataset):
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


def test_dataset_get_data(dataset):
    # 2D tensor
    input_tensor = dataset._get_data(dataset._inputs[0])

    assert isinstance(input_tensor, torch.Tensor)
    assert input_tensor.shape == torch.Size((1, 1, 32, 32))

    # Not a .pt file
    with pytest.raises(ValueError):
        tensor = dataset._get_data(dataset._inputs[1])

    # 3D instead of 2D
    with pytest.raises(ValueError):
        tensor = dataset._get_data(dataset._inputs[2])


# -------------------- DATASET 2D -------------------- #

def test_dataset_getitem_2d():
    dataset = dataset_to_modify(
        training_type="supervised", file_type="tensor", lazy_loading=True,
        input_dim=2, output_dim=2
    )

    # 2D tensor
    input_tensor, target_tensor = dataset[0]

    assert isinstance(input_tensor, torch.Tensor)
    assert input_tensor.shape == torch.Size((1, 1, 32, 32))

    assert isinstance(target_tensor, torch.Tensor)
    assert target_tensor.shape == torch.Size((1, 1, 32, 32))


# -------------------- DATASET 3D -------------------- #

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
