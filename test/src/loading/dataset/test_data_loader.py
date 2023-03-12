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

from src.loading.dataset.data_loader.data_loader import DataLoader
from src.loading.dataset.data_loader import ImageLoader, NumpyLoader, ZSTDLoader, TensorLoader


# -------------------- CONSTANT -------------------- #

DATA_PATHS = {
    "image": {
        "2D": os.path.join(paths.TEST_DATA_PATH, "data_dim", "2D", "image", "input.png"),
        "2D_modalities": os.path.join(paths.TEST_DATA_PATH, "data_dim", "2D_modalities", "image", "input.png"),
        "3D": os.path.join(paths.TEST_DATA_PATH, "data_dim", "3D", "image", "input.png"),
        "3D_modalities": os.path.join(paths.TEST_DATA_PATH, "data_dim", "3D_modalities", "image", "input.png"),
    },
    "numpy": {
        "2D": os.path.join(paths.TEST_DATA_PATH, "data_dim", "2D", "numpy", "input.npy"),
        "2D_modalities": os.path.join(paths.TEST_DATA_PATH, "data_dim", "2D_modalities", "numpy", "input.npy"),
        "3D": os.path.join(paths.TEST_DATA_PATH, "data_dim", "3D", "numpy", "input.npy"),
        "3D_modalities": os.path.join(paths.TEST_DATA_PATH, "data_dim", "3D_modalities", "numpy", "input.npy"),
    },
    "zstd": {
        "2D": os.path.join(paths.TEST_DATA_PATH, "data_dim", "2D", "zstd", "input.npz"),
        "2D_modalities": os.path.join(paths.TEST_DATA_PATH, "data_dim", "2D_modalities", "zstd", "input.npz"),
        "3D": os.path.join(paths.TEST_DATA_PATH, "data_dim", "3D", "zstd", "input.npz"),
        "3D_modalities": os.path.join(paths.TEST_DATA_PATH, "data_dim", "3D_modalities", "zstd", "input.npz"),
    },
    "tensor": {
        "2D": os.path.join(paths.TEST_DATA_PATH, "data_dim", "2D", "tensor", "input.pt"),
        "2D_modalities": os.path.join(paths.TEST_DATA_PATH, "data_dim", "2D_modalities", "tensor", "input.pt"),
        "3D": os.path.join(paths.TEST_DATA_PATH, "data_dim", "3D", "tensor", "input.pt"),
        "3D_modalities": os.path.join(paths.TEST_DATA_PATH, "data_dim", "3D_modalities", "tensor", "input.pt"),
    },
}


# -------------------- FIXTURES -------------------- #

@pytest.fixture(scope="function")
def data_loader():
    return DataLoader()


@pytest.fixture(scope="function")
def image_loader():
    return ImageLoader()


@pytest.fixture(scope="function")
def numpy_loader():
    return NumpyLoader()


@pytest.fixture(scope="function")
def zstd_loader():
    return ZSTDLoader()


@pytest.fixture(scope="function")
def tensor_loader():
    return TensorLoader()


# -------------------- DATA LOADER -------------------- #

def test_data_loader(data_loader):
    with pytest.raises(TypeError):
        data_loader._verify_path(DATA_PATHS["image"]["2D"])

    with pytest.raises(NotImplementedError):
        data_loader._load(DATA_PATHS["image"]["2D"])

    with pytest.raises(TypeError):
        data_loader(DATA_PATHS["image"]["2D"])


# -------------------- IMAGE LOADER -------------------- #

def test_image_loader_is_valid_path(image_loader):
    # Png, Jpg, Jpeg
    image_loader._verify_path(DATA_PATHS["image"]["2D"])

    # Npy
    with pytest.raises(ValueError):
        image_loader._verify_path(DATA_PATHS["numpy"]["2D"])

    # Npz
    with pytest.raises(ValueError):
        image_loader._verify_path(DATA_PATHS["zstd"]["2D"])

    # Pt
    with pytest.raises(ValueError):
        image_loader._verify_path(DATA_PATHS["tensor"]["2D"])


def test_image_loader_load(image_loader):
    # 2D
    tensor = image_loader._load(DATA_PATHS["image"]["2D"])

    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.shape == torch.Size((32, 32))

    # 2D with Modalities
    tensor = image_loader._load(DATA_PATHS["image"]["2D_modalities"])

    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.shape == torch.Size((32, 32, 3))


def test_image_loader(image_loader):
    # Png, Jpg, Jpeg -> 2D
    tensor = image_loader(DATA_PATHS["image"]["2D"])

    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.shape == torch.Size((32, 32))

    # Png, Jpg, Jpeg -> 2D with modalities
    tensor = image_loader(DATA_PATHS["image"]["2D_modalities"])

    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.shape == torch.Size((32, 32, 3))

    # Npy
    with pytest.raises(ValueError):
        image_loader(DATA_PATHS["numpy"]["2D"])

    # Npz
    with pytest.raises(ValueError):
        image_loader(DATA_PATHS["zstd"]["2D"])

    # Pt
    with pytest.raises(ValueError):
        image_loader(DATA_PATHS["tensor"]["2D"])


# -------------------- NUMPY LOADER -------------------- #

def test_numpy_loader_verify_path(numpy_loader):
    # Png, Jpg, Jpeg
    with pytest.raises(ValueError):
        numpy_loader._verify_path(DATA_PATHS["image"]["2D"])

    # Npy
    numpy_loader._verify_path(DATA_PATHS["numpy"]["2D"])

    # Npz
    with pytest.raises(ValueError):
        numpy_loader._verify_path(DATA_PATHS["zstd"]["2D"])

    # Pt
    with pytest.raises(ValueError):
        numpy_loader._verify_path(DATA_PATHS["tensor"]["2D"])


def test_numpy_loader_load(numpy_loader):
    # 2D
    tensor = numpy_loader._load(DATA_PATHS["numpy"]["2D"])

    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.shape == torch.Size((32, 32))

    # 2D with Modalities
    tensor = numpy_loader._load(DATA_PATHS["numpy"]["2D_modalities"])

    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.shape == torch.Size((32, 32, 3))

    # 3D
    tensor = numpy_loader._load(DATA_PATHS["numpy"]["3D"])

    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.shape == torch.Size((32, 32, 32))

    # 3D with Modalities
    tensor = numpy_loader._load(DATA_PATHS["numpy"]["3D_modalities"])

    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.shape == torch.Size((5, 32, 32, 32))


def test_numpy_loader(numpy_loader):
    # Npy -> 2D
    tensor = numpy_loader(DATA_PATHS["numpy"]["2D"])

    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.shape == torch.Size((32, 32))

    # Npy -> 2D with Modalities
    tensor = numpy_loader(DATA_PATHS["numpy"]["2D_modalities"])

    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.shape == torch.Size((32, 32, 3))

    # Npy -> 3D
    tensor = numpy_loader(DATA_PATHS["numpy"]["3D"])

    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.shape == torch.Size((32, 32, 32))

    # Npy -> 3D with Modalities
    tensor = numpy_loader(DATA_PATHS["numpy"]["3D_modalities"])

    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.shape == torch.Size((5, 32, 32, 32))

    # Png, Jpg, Jpeg
    with pytest.raises(ValueError):
        numpy_loader(DATA_PATHS["image"]["2D"])

    # Npz
    with pytest.raises(ValueError):
        numpy_loader(DATA_PATHS["zstd"]["2D"])

    # Pt
    with pytest.raises(ValueError):
        numpy_loader(DATA_PATHS["tensor"]["2D"])


# -------------------- ZSTD LOADER -------------------- #

def test_zstd_loader_verify_path(zstd_loader):
    # Png, Jpg, Jpeg
    with pytest.raises(ValueError):
        zstd_loader._verify_path(DATA_PATHS["image"]["2D"])

    # Npy
    with pytest.raises(ValueError):
        zstd_loader._verify_path(DATA_PATHS["numpy"]["2D"])

    # Npz
    zstd_loader._verify_path(DATA_PATHS["zstd"]["2D"])

    # Pt
    with pytest.raises(ValueError):
        zstd_loader._verify_path(DATA_PATHS["tensor"]["2D"])


def test_zstd_loader_load(zstd_loader):
    # 2D
    tensor = zstd_loader._load(DATA_PATHS["zstd"]["2D"])

    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.shape == torch.Size((32, 32))

    # 2D with Modalities
    tensor = zstd_loader._load(DATA_PATHS["zstd"]["2D_modalities"])

    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.shape == torch.Size((32, 32, 3))

    # 3D
    tensor = zstd_loader._load(DATA_PATHS["zstd"]["3D"])

    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.shape == torch.Size((32, 32, 32))

    # 3D with Modalities
    tensor = zstd_loader._load(DATA_PATHS["zstd"]["3D_modalities"])

    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.shape == torch.Size((5, 32, 32, 32))


def test_zstd_loader(zstd_loader):
    # 2D
    tensor = zstd_loader(DATA_PATHS["zstd"]["2D"])

    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.shape == torch.Size((32, 32))

    # 2D with Modalities
    tensor = zstd_loader(DATA_PATHS["zstd"]["2D_modalities"])

    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.shape == torch.Size((32, 32, 3))

    # 3D
    tensor = zstd_loader(DATA_PATHS["zstd"]["3D"])

    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.shape == torch.Size((32, 32, 32))

    # 3D with Modalities
    tensor = zstd_loader(DATA_PATHS["zstd"]["3D_modalities"])

    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.shape == torch.Size((5, 32, 32, 32))

    # Png, Jpg, Jpeg
    with pytest.raises(ValueError):
        zstd_loader(DATA_PATHS["image"]["2D"])

    # Npy
    with pytest.raises(ValueError):
        zstd_loader(DATA_PATHS["numpy"]["2D"])

    # Pt
    with pytest.raises(ValueError):
        zstd_loader(DATA_PATHS["tensor"]["2D"])


# -------------------- TENSOR LOADER -------------------- #

def test_tensor_loader_verify_path(tensor_loader):
    # Png, Jpg, Jpeg
    with pytest.raises(ValueError):
        tensor_loader._verify_path(DATA_PATHS["image"]["2D"])

    # Npy
    with pytest.raises(ValueError):
        tensor_loader._verify_path(DATA_PATHS["numpy"]["2D"])

    # Npz
    with pytest.raises(ValueError):
        tensor_loader._verify_path(DATA_PATHS["zstd"]["2D"])

    # Pt
    tensor_loader._verify_path(DATA_PATHS["tensor"]["2D"])


def test_tensor_loader_load(tensor_loader):
    # 2D
    tensor = tensor_loader._load(DATA_PATHS["tensor"]["2D"])

    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.shape == torch.Size((32, 32))

    # 2D with Modalities
    tensor = tensor_loader._load(DATA_PATHS["tensor"]["2D_modalities"])

    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.shape == torch.Size((32, 32, 3))

    # 3D
    tensor = tensor_loader._load(DATA_PATHS["tensor"]["3D"])

    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.shape == torch.Size((32, 32, 32))

    # 3D with Modalities
    tensor = tensor_loader._load(DATA_PATHS["tensor"]["3D_modalities"])

    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.shape == torch.Size((5, 32, 32, 32))


def test_tensor_loader(tensor_loader):
    # 2D
    tensor = tensor_loader(DATA_PATHS["tensor"]["2D"])

    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.shape == torch.Size((32, 32))

    # 2D with Modalities
    tensor = tensor_loader(DATA_PATHS["tensor"]["2D_modalities"])

    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.shape == torch.Size((32, 32, 3))

    # 3D
    tensor = tensor_loader(DATA_PATHS["tensor"]["3D"])

    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.shape == torch.Size((32, 32, 32))

    # 3D with Modalities
    tensor = tensor_loader(DATA_PATHS["tensor"]["3D_modalities"])

    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.shape == torch.Size((5, 32, 32, 32))

    # Png, Jpg, Jpeg
    with pytest.raises(ValueError):
        tensor_loader(DATA_PATHS["image"]["2D"])

    # Npy
    with pytest.raises(ValueError):
        tensor_loader(DATA_PATHS["numpy"]["2D"])

    # Npz
    with pytest.raises(ValueError):
        tensor_loader(DATA_PATHS["zstd"]["2D"])
