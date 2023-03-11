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

from src.loading.loader.loading_manager import Loading
from src.loading.loader import LoadingUnsupervised, LoadingSupervised

from src.loading.dataset import DataSet2D, DataSet3D


# -------------------- CONSTANT -------------------- #

DATA_PATHS = {
    "depth_0": os.path.join(paths.TEST_PATH, "data_depth_0"),
    "depth_1": os.path.join(paths.TEST_PATH, "data_depth_1"),
    "depth_2": os.path.join(paths.TEST_PATH, "data_depth_2")
}

LENGHT_TENSOR_3D = 32


# -------------------- FIXTURES -------------------- #

@pytest.fixture(scope="function")
def loading():
    return Loading(
        params={
            "training_type": "supervised", "file_type": "tensor", "lazy_loading": True,
            "input_dim": 2, "output_dim": 2
        }
    )


@pytest.fixture(scope="function")
def unsupervised_loading():
    return Loading(
        params={
            "training_type": "supervised", "file_type": "tensor", "lazy_loading": True,
            "input_dim": 2, "output_dim": 2
        }
    )


@pytest.fixture(scope="function")
def supervised_loading():
    return Loading(
        params={
            "training_type": "supervised", "file_type": "tensor", "lazy_loading": True,
            "input_dim": 2, "output_dim": 2
        }
    )


# -------------------- LOADING -------------------- #

def test_loading(loading):
    with pytest.raises(NotImplementedError):
        loading._order_paths([])

    with pytest.raises(NotImplementedError):
        loading._generate_dataset()


# -------------------- UNSUPERVISED LOADING -------------------- #

def test_unsupervised_loading():
    pass


# -------------------- UNSUPERVISED LOADING -------------------- #

def test_supervised_loading():
    pass
