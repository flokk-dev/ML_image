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

from src.loading import Loader, UnsupervisedLoader, SupervisedLoader

from src.loading.data_loader import UnsupervisedDataLoader, SupervisedDataLoader


# -------------------- CONSTANT -------------------- #

DATA_PATHS = {
    "depth_0": os.path.join(paths.TEST_DATA_PATH, "data_depth_0"),
    "depth_1": os.path.join(paths.TEST_DATA_PATH, "data_depth_1"),
    "depth_2": os.path.join(paths.TEST_DATA_PATH, "data_depth_2")
}

LENGHT_TENSOR_3D = 32


# -------------------- FIXTURES -------------------- #

@pytest.fixture(scope="function")
def loading():
    return Loader(
        params={
            "file_type": "tensor", "lazy_loading": True,
            "file_depth": 0, "dataset_name": "data",
            "input_dim": 2, "output_dim": 2, "out_channels": 1,
            "batch_size": 4
        }
    )


@pytest.fixture(scope="function")
def unsupervised_loading():
    return UnsupervisedLoader(
        params={
            "file_type": "tensor", "lazy_loading": True,
            "file_depth": 0, "dataset_name": "data",
            "input_dim": 2, "output_dim": 2, "out_channels": 1,
            "batch_size": 4
        }
    )


@pytest.fixture(scope="function")
def supervised_loading():
    return SupervisedLoader(
        params={
            "file_type": "tensor", "lazy_loading": True,
            "file_depth": 0, "dataset_name": "data",
            "input_dim": 2, "output_dim": 2, "out_channels": 1,
            "batch_size": 4
        }
    )


# -------------------- LOADING -------------------- #

def test_loading(loading):
    with pytest.raises(NotImplementedError):
        loading._generate_data_loaders()


def test_loading_file_depth(loading):
    # depth = 0
    depth = loading._file_depth(os.path.join("data", "1"), "data")
    assert depth == 0

    depth = loading._file_depth(os.path.join("0", "data", "1"), "data")
    assert depth == 0

    # depth = 1
    depth = loading._file_depth(os.path.join("data", "1", "2"), "data")
    assert depth == 1

    depth = loading._file_depth(os.path.join("0", "data", "1", "2"), "data")
    assert depth == 1

    # depth = 2
    depth = loading._file_depth(os.path.join("data", "1", "2", "3"), "data")
    assert depth == 2

    depth = loading._file_depth(os.path.join("0", "data", "1", "2", "3"), "data")
    assert depth == 2


def test_loading_extract_paths_0(loading):
    loading._params["file_depth"] = 0
    loading._params["dataset_name"] = "data_depth_0"

    file_paths = loading._extract_paths(DATA_PATHS["depth_0"])
    assert len(file_paths) == 20

    assert os.path.basename(file_paths[0]) == "0_input.pt"
    assert os.path.basename(file_paths[1]) == "0_target.pt"

    assert os.path.basename(file_paths[-2]) == "9_input.pt"
    assert os.path.basename(file_paths[-1]) == "9_target.pt"


def test_loading_extract_paths_1(loading):
    loading._params["file_depth"] = 1
    loading._params["dataset_name"] = "data_depth_1"

    file_paths = loading._extract_paths(DATA_PATHS["depth_1"])
    assert len(file_paths) == 20

    assert os.path.join(
        file_paths[0].split(os.sep)[-2], os.path.basename(file_paths[0])
    ) == os.path.join("0", "input.pt")
    assert os.path.join(
        file_paths[1].split(os.sep)[-2], os.path.basename(file_paths[1])
    ) == os.path.join("0", "target.pt")

    assert os.path.join(
        file_paths[-2].split(os.sep)[-2], os.path.basename(file_paths[-2])
    ) == os.path.join("9", "input.pt")
    assert os.path.join(
        file_paths[-1].split(os.sep)[-2], os.path.basename(file_paths[-1])
    ) == os.path.join("9", "target.pt")


def test_loading_extract_paths_2(loading):
    loading._params["file_depth"] = 2
    loading._params["dataset_name"] = "data_depth_2"

    file_paths = loading._extract_paths(DATA_PATHS["depth_2"])
    assert len(file_paths) == 20

    assert os.path.join(
        file_paths[0].split(os.sep)[-2], os.path.basename(file_paths[0])
    ) == os.path.join("0", "input.pt")
    assert os.path.join(
        file_paths[1].split(os.sep)[-2], os.path.basename(file_paths[1])
    ) == os.path.join("0", "target.pt")

    assert os.path.join(
        file_paths[-2].split(os.sep)[-2], os.path.basename(file_paths[-2])
    ) == os.path.join("9", "input.pt")
    assert os.path.join(
        file_paths[-1].split(os.sep)[-2], os.path.basename(file_paths[-1])
    ) == os.path.join("9", "target.pt")


# -------------------- UNSUPERVISED LOADING -------------------- #

def test_unsupervised_loading_extract_paths_0(unsupervised_loading):
    unsupervised_loading._params["file_depth"] = 0
    unsupervised_loading._params["dataset_name"] = "data_depth_0"

    unsupervised_loading._extract_paths(DATA_PATHS["depth_0"])
    assert len(unsupervised_loading._input_paths) == 2

    # TRAIN
    assert len(unsupervised_loading._input_paths["train"]) == 16

    assert os.path.basename(unsupervised_loading._input_paths["train"][0]) == "0_input.pt"
    assert os.path.basename(unsupervised_loading._input_paths["train"][1]) == "0_target.pt"

    assert os.path.basename(unsupervised_loading._input_paths["train"][-2]) == "7_input.pt"
    assert os.path.basename(unsupervised_loading._input_paths["train"][-1]) == "7_target.pt"

    # VALID
    assert len(unsupervised_loading._input_paths["valid"]) == 4

    assert os.path.basename(unsupervised_loading._input_paths["valid"][0]) == "8_input.pt"
    assert os.path.basename(unsupervised_loading._input_paths["valid"][1]) == "8_target.pt"

    assert os.path.basename(unsupervised_loading._input_paths["valid"][-2]) == "9_input.pt"
    assert os.path.basename(unsupervised_loading._input_paths["valid"][-1]) == "9_target.pt"


def test_unsupervised_loading_extract_paths_1(unsupervised_loading):
    unsupervised_loading._params["file_depth"] = 1
    unsupervised_loading._params["dataset_name"] = "data_depth_1"

    unsupervised_loading._extract_paths(DATA_PATHS["depth_1"])
    assert len(unsupervised_loading._input_paths) == 2

    # TRAIN
    assert len(unsupervised_loading._input_paths["train"]) == 16

    assert os.path.join(
        unsupervised_loading._input_paths["train"][0].split(os.sep)[-2],
        os.path.basename(unsupervised_loading._input_paths["train"][0])
    ) == os.path.join("0", "input.pt")
    assert os.path.join(
        unsupervised_loading._input_paths["train"][1].split(os.sep)[-2],
        os.path.basename(unsupervised_loading._input_paths["train"][1])
    ) == os.path.join("0", "target.pt")

    assert os.path.join(
        unsupervised_loading._input_paths["train"][-2].split(os.sep)[-2],
        os.path.basename(unsupervised_loading._input_paths["train"][-2])
    ) == os.path.join("7", "input.pt")
    assert os.path.join(
        unsupervised_loading._input_paths["train"][-1].split(os.sep)[-2],
        os.path.basename(unsupervised_loading._input_paths["train"][-1])
    ) == os.path.join("7", "target.pt")

    # VALID
    assert len(unsupervised_loading._input_paths["valid"]) == 4

    assert os.path.join(
        unsupervised_loading._input_paths["valid"][0].split(os.sep)[-2],
        os.path.basename(unsupervised_loading._input_paths["valid"][0])
    ) == os.path.join("8", "input.pt")
    assert os.path.join(
        unsupervised_loading._input_paths["valid"][1].split(os.sep)[-2],
        os.path.basename(unsupervised_loading._input_paths["valid"][1])
    ) == os.path.join("8", "target.pt")

    assert os.path.join(
        unsupervised_loading._input_paths["valid"][-2].split(os.sep)[-2],
        os.path.basename(unsupervised_loading._input_paths["valid"][-2])
    ) == os.path.join("9", "input.pt")
    assert os.path.join(
        unsupervised_loading._input_paths["valid"][-1].split(os.sep)[-2],
        os.path.basename(unsupervised_loading._input_paths["valid"][-1])
    ) == os.path.join("9", "target.pt")


def test_unsupervised_loading_call(unsupervised_loading):
    unsupervised_loading._params["file_depth"] = 0
    unsupervised_loading._params["dataset_name"] = "data_depth_0"

    batch_size = unsupervised_loading._params["batch_size"]

    data_loaders = unsupervised_loading(DATA_PATHS["depth_0"])
    assert isinstance(data_loaders, dict)

    for step, data_loader in data_loaders.items():
        assert isinstance(data_loader, UnsupervisedDataLoader)

        for sub_loader in data_loader:
            for inputs, in sub_loader:
                assert isinstance(inputs, torch.Tensor)
                assert inputs.dtype == torch.float32

                assert inputs.shape == torch.Size((batch_size, 1, 32, 32))


# -------------------- UNSUPERVISED LOADING -------------------- #

def test_supervised_loading_extract_paths_0(supervised_loading):
    supervised_loading._params["file_depth"] = 0
    supervised_loading._params["dataset_name"] = "data_depth_0"

    supervised_loading._extract_paths(DATA_PATHS["depth_0"])
    assert len(supervised_loading._input_paths) == 2
    assert len(supervised_loading._target_paths) == 2

    # TRAIN
    assert len(supervised_loading._input_paths["train"]) == 8
    assert len(supervised_loading._target_paths["train"]) == 8

    assert os.path.basename(supervised_loading._input_paths["train"][0]) == "0_input.pt"
    assert os.path.basename(supervised_loading._input_paths["train"][1]) == "1_input.pt"

    assert os.path.basename(supervised_loading._target_paths["train"][0]) == "0_target.pt"
    assert os.path.basename(supervised_loading._target_paths["train"][1]) == "1_target.pt"

    # VALID
    assert len(supervised_loading._input_paths["valid"]) == 2
    assert len(supervised_loading._target_paths["valid"]) == 2

    assert os.path.basename(supervised_loading._input_paths["valid"][-2]) == "8_input.pt"
    assert os.path.basename(supervised_loading._input_paths["valid"][-1]) == "9_input.pt"

    assert os.path.basename(supervised_loading._target_paths["valid"][-2]) == "8_target.pt"
    assert os.path.basename(supervised_loading._target_paths["valid"][-1]) == "9_target.pt"


def test_supervised_loading_extract_paths_1(supervised_loading):
    supervised_loading._params["file_depth"] = 1
    supervised_loading._params["dataset_name"] = "data_depth_1"

    supervised_loading._extract_paths(DATA_PATHS["depth_1"])
    assert len(supervised_loading._input_paths) == 2
    assert len(supervised_loading._target_paths) == 2

    # TRAIN
    assert len(supervised_loading._input_paths["train"]) == 8
    assert len(supervised_loading._target_paths["train"]) == 8

    assert os.path.join(
        supervised_loading._input_paths["train"][0].split(os.sep)[-2],
        os.path.basename(supervised_loading._input_paths["train"][0])
    ) == os.path.join("0", "input.pt")
    assert os.path.join(
        supervised_loading._input_paths["train"][1].split(os.sep)[-2],
        os.path.basename(supervised_loading._input_paths["train"][1])
    ) == os.path.join("1", "input.pt")

    assert os.path.join(
        supervised_loading._target_paths["train"][0].split(os.sep)[-2],
        os.path.basename(supervised_loading._target_paths["train"][-2])
    ) == os.path.join("0", "target.pt")
    assert os.path.join(
        supervised_loading._target_paths["train"][1].split(os.sep)[-2],
        os.path.basename(supervised_loading._target_paths["train"][-1])
    ) == os.path.join("1", "target.pt")

    # VALID
    assert len(supervised_loading._input_paths["valid"]) == 2
    assert len(supervised_loading._target_paths["valid"]) == 2

    assert os.path.join(
        supervised_loading._input_paths["valid"][0].split(os.sep)[-2],
        os.path.basename(supervised_loading._input_paths["valid"][0])
    ) == os.path.join("8", "input.pt")
    assert os.path.join(
        supervised_loading._input_paths["valid"][1].split(os.sep)[-2],
        os.path.basename(supervised_loading._input_paths["valid"][1])
    ) == os.path.join("9", "input.pt")

    assert os.path.join(
        supervised_loading._target_paths["valid"][0].split(os.sep)[-2],
        os.path.basename(supervised_loading._target_paths["valid"][-2])
    ) == os.path.join("8", "target.pt")
    assert os.path.join(
        supervised_loading._target_paths["valid"][1].split(os.sep)[-2],
        os.path.basename(supervised_loading._target_paths["valid"][-1])
    ) == os.path.join("9", "target.pt")


def test_supervised_loading_call(supervised_loading):
    supervised_loading._params["file_depth"] = 0
    supervised_loading._params["dataset_name"] = "data_depth_0"

    batch_size = supervised_loading._params["batch_size"]

    data_loaders = supervised_loading(DATA_PATHS["depth_0"])
    assert isinstance(data_loaders, dict)

    for step, data_loader in data_loaders.items():
        assert isinstance(data_loader, SupervisedDataLoader)

        for sub_loader in data_loader:
            for inputs, targets in sub_loader:
                assert isinstance(inputs, torch.Tensor)
                assert inputs.dtype == torch.float32

                assert isinstance(targets, torch.Tensor)
                assert targets.dtype == torch.float32

                assert inputs.shape == torch.Size((batch_size, 1, 32, 32))
                assert targets.shape == torch.Size((batch_size, 1, 32, 32))
