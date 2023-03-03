"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import os

# IMPORT: test
import pytest

# IMPORT: deep learning
import torch

# IMPORT: project
import paths
from src.loading.dataset import DataSet, DataSet2D, DataSet3D


"""
CONSTANT
"""

DICO = {
    "lazy_loading": True,
    "data_augmentation": False,
    "nb_data": 3,
    "patch_height": 5,
    "batch_size": 32,
    "epochs": 25,
    "workers": 2,
    "lr": 1e-2,
    "lr_multiplier": 0.75,
    "valid_coeff": 0.2,
}


"""
TEST
"""


# DATASET
@pytest.fixture(scope="function")
def dataset():
    dataset = DataSet(DICO, DATA_DICO, INPUT_PATH, TARGET_PATH)
    return dataset


def test_load_volume(dataset):
    volume = dataset._load_volume(INPUT_PATH[0], DATA_DICO[9]["shape"])

    assert volume.shape == torch.Size([1, 103, 512, 512])
    assert isinstance(volume, torch.Tensor)

    gc.collect()


# DATASET 2D
@pytest.fixture(scope="function")
def dataset2D():
    dataset2D = DataSet2D(DICO, DATA_DICO, INPUT_PATH, TARGET_PATH)
    return dataset2D


def test_my_collate_dataset2D(dataset2D):
    input_volume = dataset2D._load_volume(INPUT_PATH[0], DATA_DICO[9]["shape"])
    target_volume = dataset2D._load_volume(TARGET_PATH[0], DATA_DICO[9]["shape"])

    input_volume, target_volume = [input_volume], [target_volume]
    assert isinstance(dataset2D._my_collate(input_volume, target_volume), tuple)

    input_volume, target_volume = dataset2D._my_collate(input_volume, target_volume)

    # INPUT
    assert isinstance(input_volume, torch.Tensor)

    input_shape = input_volume.shape
    assert len(input_shape) == 4

    assert input_shape[1:] == torch.Size([1, 512, 512])

    # TARGET
    assert isinstance(target_volume, torch.Tensor)

    target_shape = target_volume.shape
    assert len(target_shape) == 4
    assert target_shape[1:] == torch.Size([1, 512, 512])

    gc.collect()


# DATASET 2.5D
@pytest.fixture(scope="function")
def dataset25D():
    dataset25D = DataSet25D(DICO, DATA_DICO, INPUT_PATH, TARGET_PATH)
    return dataset25D


def test_my_collate_dataset25D(dataset25D):
    input_volume = dataset25D._load_volume(INPUT_PATH[0], DATA_DICO[9]["shape"])
    target_volume = dataset25D._load_volume(TARGET_PATH[0], DATA_DICO[9]["shape"])

    input_volume, target_volume = [input_volume], [target_volume]
    assert isinstance(dataset25D._my_collate(input_volume, target_volume), tuple)

    input_volume, target_volume = dataset25D._my_collate(input_volume, target_volume)

    # INPUT
    assert isinstance(input_volume, torch.Tensor)

    input_shape = input_volume.shape
    assert len(input_shape) == 4

    assert input_shape[1:] == torch.Size([5, 512, 512])

    # TARGET
    assert isinstance(target_volume, torch.Tensor)

    target_shape = target_volume.shape
    assert len(target_shape) == 4

    assert target_shape[1:] == torch.Size([1, 512, 512])

    gc.collect()


# DATASET 3D
@pytest.fixture(scope="function")
def dataset3D():
    dataset3D = DataSet3D(DICO, DATA_DICO, INPUT_PATH, TARGET_PATH)
    return dataset3D


def test_my_collate_dataset3D(dataset3D):
    input_volume = dataset3D._load_volume(INPUT_PATH[0], DATA_DICO[9]["shape"])
    target_volume = dataset3D._load_volume(TARGET_PATH[0], DATA_DICO[9]["shape"])

    input_volume, target_volume = [input_volume], [target_volume]
    assert isinstance(dataset3D._my_collate(input_volume, target_volume), tuple)

    input_volume, target_volume = dataset3D._my_collate(input_volume, target_volume)

    # INPUT
    assert isinstance(input_volume, torch.Tensor)

    input_shape = input_volume.shape
    assert len(input_shape) == 5
    assert input_shape[1:] == torch.Size([1, 64, 64, 64])

    # TARGET
    assert isinstance(target_volume, torch.Tensor)

    target_shape = target_volume.shape
    assert len(target_shape) == 5

    assert target_shape[1:] == torch.Size([1, 64, 64, 64])

    gc.collect()
