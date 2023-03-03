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
from src.loading.dataset.data_loader import DataLoader, \
    ImageLoader, NumpyLoader, ZSTDLoader, TensorLoader


# -------------------- DATA LOADER -------------------- #

def test_data_loader():
    image_path = os.path.join(paths.DATA_TEST_PATH, "2D", "png", "input.png")

    data_loader = DataLoader()
    with pytest.raises(TypeError):
        data_loader._verify_path(image_path)

    with pytest.raises(NotImplementedError):
        data_loader._load(image_path)

    with pytest.raises(TypeError):
        data_loader(image_path)


# -------------------- IMAGE LOADER -------------------- #

def test_image_loader_verify_path():
    image_loader = ImageLoader()

    # Png, Jpg, Jpeg
    image_path = os.path.join(paths.DATA_TEST_PATH, "2D", "png", "input.png")
    image_loader._verify_path(image_path)

    # Npy
    numpy_path = os.path.join(paths.DATA_TEST_PATH, "2D", "npy", "input.npy")
    with pytest.raises(ValueError):
        image_loader._verify_path(numpy_path)

    # Npz
    zstd_path = os.path.join(paths.DATA_TEST_PATH, "2D", "npz", "input.npz")
    with pytest.raises(ValueError):
        image_loader._verify_path(zstd_path)

    # Pt
    tensor_path = os.path.join(paths.DATA_TEST_PATH, "2D", "pt", "input.pt")
    with pytest.raises(ValueError):
        image_loader._verify_path(tensor_path)


def test_image_loader_load():
    image_loader = ImageLoader()

    # 2D
    image_path = os.path.join(paths.DATA_TEST_PATH, "2D", "png", "input.png")
    tensor = image_loader._load(image_path)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.shape == torch.Size((32, 32))

    # 2D with Modalities
    image_path = os.path.join(paths.DATA_TEST_PATH, "2D_modalities", "png", "input.png")
    tensor = image_loader._load(image_path)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.shape == torch.Size((32, 32, 3))


def test_image_loader():
    image_loader = ImageLoader()

    # Png, Jpg, Jpeg -> 2D
    image_path = os.path.join(paths.DATA_TEST_PATH, "2D", "png", "input.png")
    tensor = image_loader(image_path)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == torch.Size((32, 32))

    # Png, Jpg, Jpeg -> 2D with modalities
    image_path = os.path.join(paths.DATA_TEST_PATH, "2D", "png", "input.png")
    tensor = image_loader(image_path)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == torch.Size((32, 32, 3))

    # Npy
    numpy_path = os.path.join(paths.DATA_TEST_PATH, "2D", "npy", "input.npy")
    with pytest.raises(ValueError):
        image_loader(numpy_path)

    # Npz
    zstd_path = os.path.join(paths.DATA_TEST_PATH, "2D", "npz", "input.npz")
    with pytest.raises(ValueError):
        image_loader(zstd_path)

    # Pt
    tensor_path = os.path.join(paths.DATA_TEST_PATH, "2D", "pt", "input.pt")
    with pytest.raises(ValueError):
        image_loader(tensor_path)


# -------------------- NUMPY LOADER -------------------- #

def test_numpy_loader_verify_path():
    numpy_loader = NumpyLoader()

    # Png, Jpg, Jpeg
    image_path = os.path.join(paths.DATA_TEST_PATH, "2D", "png", "input.png")
    with pytest.raises(ValueError):
        numpy_loader._verify_path(image_path)

    # Npy
    numpy_path = os.path.join(paths.DATA_TEST_PATH, "2D", "npy", "input.npy")
    numpy_loader._verify_path(numpy_path)

    # Npz
    zstd_path = os.path.join(paths.DATA_TEST_PATH, "2D", "npz", "input.npz")
    with pytest.raises(ValueError):
        numpy_loader._verify_path(zstd_path)

    # Pt
    tensor_path = os.path.join(paths.DATA_TEST_PATH, "2D", "pt", "input.pt")
    with pytest.raises(ValueError):
        numpy_loader._verify_path(tensor_path)


def test_numpy_loader_load():
    numpy_loader = NumpyLoader()

    # 2D
    numpy_path = os.path.join(paths.DATA_TEST_PATH, "2D", "npy", "input.npy")
    tensor = numpy_loader._load(numpy_path)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.shape == torch.Size((32, 32))

    # 2D with Modalities
    numpy_path = os.path.join(paths.DATA_TEST_PATH, "2D_modalities", "npy", "input.npy")
    tensor = numpy_loader._load(numpy_path)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.shape == torch.Size((32, 32, 3))

    # 3D with Modalities
    numpy_path = os.path.join(paths.DATA_TEST_PATH, "2D_modalities", "npy", "input.npy")
    tensor = numpy_loader._load(numpy_path)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.shape == torch.Size((32, 32, 3))


def test_numpy_loader():
    numpy_loader = NumpyLoader()

    # Npy -> 2D
    numpy_path = os.path.join(paths.DATA_TEST_PATH, "2D", "npy", "input.npy")
    tensor = numpy_loader._load(numpy_path)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.shape == torch.Size((32, 32))

    # Npy -> 2D with Modalities
    numpy_path = os.path.join(paths.DATA_TEST_PATH, "2D_modalities", "npy", "input.npy")
    tensor = numpy_loader._load(numpy_path)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.shape == torch.Size((32, 32, 3))

    # Png, Jpg, Jpeg
    image_path = os.path.join(paths.DATA_TEST_PATH, "2D", "png", "input.png")
    with pytest.raises(ValueError):
        numpy_loader._verify_path(image_path)

    # Npz
    zstd_path = os.path.join(paths.DATA_TEST_PATH, "2D", "npz", "input.npz")
    with pytest.raises(ValueError):
        numpy_loader(zstd_path)

    # Pt
    tensor_path = os.path.join(paths.DATA_TEST_PATH, "2D", "pt", "input.pt")
    with pytest.raises(ValueError):
        numpy_loader(tensor_path)


# -------------------- ZSTD LOADER -------------------- #

def test_zstd_loader():
    zstd_loader = ZSTDLoader()

    # Png, Jpg, Jpeg
    image_path = os.path.join(paths.DATA_TEST_PATH, "2D", "png", "input.png")
    with pytest.raises(ValueError):
        zstd_loader(image_path)

    # Npy
    numpy_path = os.path.join(paths.DATA_TEST_PATH, "2D", "npy", "input.npy")
    with pytest.raises(ValueError):
        zstd_loader(numpy_path)

    # Npz
    zstd_path = os.path.join(paths.DATA_TEST_PATH, "2D", "npz", "input.npz")
    assert isinstance(zstd_loader(zstd_path), torch.Tensor)

    # Pt
    tensor_path = os.path.join(paths.DATA_TEST_PATH, "2D", "pt", "input.pt")
    with pytest.raises(ValueError):
        zstd_loader(tensor_path)


# -------------------- TENSOR LOADER -------------------- #

def test_tensor_loader():
    tensor_loader = TensorLoader()

    # Png, Jpg, Jpeg
    image_path = os.path.join(paths.DATA_TEST_PATH, "2D", "png", "input.png")
    with pytest.raises(ValueError):
        tensor_loader(image_path)

    # Npy
    numpy_path = os.path.join(paths.DATA_TEST_PATH, "2D", "npy", "input.npy")
    with pytest.raises(ValueError):
        tensor_loader(numpy_path)

    # Npz
    zstd_path = os.path.join(paths.DATA_TEST_PATH, "2D", "npz", "input.npz")
    with pytest.raises(ValueError):
        tensor_loader(zstd_path)

    # Pt
    tensor_path = os.path.join(paths.DATA_TEST_PATH, "2D", "pt", "input.pt")
    assert isinstance(tensor_loader(tensor_path), torch.Tensor)
