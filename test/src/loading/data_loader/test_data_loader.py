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

from src.loading.data_loader import DataLoader, \
    UnsupervisedDataLoader, SupervisedDataLoader

from src.loading.dataset import UnsupervisedDataSet, SupervisedDataSet


# -------------------- CONSTANT -------------------- #

DATA_PATHS = {
    "2D": os.path.join(paths.TEST_DATA_PATH, "data_dim", "2D", "tensor", "input.pt"),
    "3D": os.path.join(paths.TEST_DATA_PATH, "data_dim", "3D", "tensor", "input.pt"),
}

LENGHT_TENSOR_3D = 32


# -------------------- FIXTURES -------------------- #

def dataset_to_modify(training_type, lazy_loading, input_dim, output_dim):
    if training_type == "unsupervised":
        return UnsupervisedDataSet(
            params={
                "file_type": "tensor", "lazy_loading": lazy_loading,
                "input_dim": input_dim, "output_dim": output_dim, "out_channels": 1,
            },
            inputs=[DATA_PATHS[f"{str(input_dim)}D"] for i in range(10)],
        )

    elif training_type == "supervised":
        return SupervisedDataSet(
            params={
                "file_type": "tensor", "lazy_loading": lazy_loading,
                "input_dim": input_dim, "output_dim": output_dim, "out_channels": 1
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


# -------------------- UNSUPERVISED DATA LOADER -------------------- #

def test_unsupervised_loader():
    dataset = dataset_to_modify(
        training_type="unsupervised", lazy_loading=True,
        input_dim=2, output_dim=2
    )

    # batch size --> from 10 to 20
    for batch_size in range(10, 21):
        lazy_loader = UnsupervisedDataLoader({"batch_size": batch_size}, dataset)

        for sub_loader in lazy_loader:
            for inputs, in sub_loader:
                assert isinstance(inputs, torch.Tensor)
                assert inputs.dtype == torch.float32

                assert inputs.shape == torch.Size((batch_size, 1, 32, 32))


# -------------------- SUPERVISED DATA LOADER -------------------- #

def test_supervised_loader():
    dataset = dataset_to_modify(
        training_type="supervised", lazy_loading=True,
        input_dim=3, output_dim=2
    )

    # batch size --> from 10 to 20
    for batch_size in range(10, 21):
        lazy_loader = SupervisedDataLoader({"batch_size": batch_size}, dataset)

        for sub_loader in lazy_loader:
            for inputs, targets in sub_loader:
                assert isinstance(inputs, torch.Tensor)
                assert inputs.dtype == torch.float32

                assert isinstance(targets, torch.Tensor)
                assert targets.dtype == torch.float32

                assert inputs.shape == torch.Size((batch_size, 1, 32, 32))
                assert targets.shape == torch.Size((batch_size, 1, 32, 32))


# -------------------- 2D DATA -------------------- #

def test_data_loader_2d():
    dataset = dataset_to_modify(
        training_type="supervised", lazy_loading=True,
        input_dim=2, output_dim=2
    )

    # batch size --> from 10 to 20
    for batch_size in range(10, 21):
        lazy_loader = SupervisedDataLoader({"batch_size": batch_size}, dataset)

        for sub_loader in lazy_loader:
            for inputs, targets in sub_loader:
                assert isinstance(inputs, torch.Tensor)
                assert inputs.dtype == torch.float32

                assert isinstance(targets, torch.Tensor)
                assert targets.dtype == torch.float32

                assert inputs.shape == torch.Size((batch_size, 1, 32, 32))
                assert targets.shape == torch.Size((batch_size, 1, 32, 32))


# -------------------- 3D DATA -------------------- #

def test_data_loader_3d_2d():
    dataset = dataset_to_modify(
        training_type="supervised", lazy_loading=True,
        input_dim=3, output_dim=2
    )

    # batch size --> from 10 to 20
    for batch_size in range(10, 21):
        lazy_loader = SupervisedDataLoader({"batch_size": batch_size}, dataset)

        for sub_loader in lazy_loader:
            for inputs, targets in sub_loader:
                assert isinstance(inputs, torch.Tensor)
                assert inputs.dtype == torch.float32

                assert isinstance(targets, torch.Tensor)
                assert targets.dtype == torch.float32

                assert inputs.shape == torch.Size((batch_size, 1, 32, 32))
                assert targets.shape == torch.Size((batch_size, 1, 32, 32))


def test_data_loader_3d_25d():
    dataset = dataset_to_modify(
        training_type="supervised", lazy_loading=True,
        input_dim=3, output_dim=2.5
    )

    # batch size --> from 10 to 20
    for batch_size in range(10, 21):
        lazy_loader = SupervisedDataLoader({"batch_size": batch_size}, dataset)

        for sub_loader in lazy_loader:
            for inputs, targets in sub_loader:
                assert isinstance(inputs, torch.Tensor)
                assert inputs.dtype == torch.float32

                assert isinstance(targets, torch.Tensor)
                assert targets.dtype == torch.float32

                assert inputs.shape == torch.Size((batch_size, 5, 32, 32))
                assert targets.shape == torch.Size((batch_size, 1, 32, 32))
