"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: test
import pytest

# IMPORT: deep learning
import torch

# IMPORT: project
from src.training.components.model import ModelManager
from src.training.components.model.unet import UNet, AttentionUNet, Transformer, SWinTransformer


# -------------------- FIXTURES -------------------- #

@pytest.fixture(scope="function")
def model_manager():
    return ModelManager()


@pytest.fixture(scope="function")
def data_info():
    return {
        "spatial_dims": 2,
        "img_size": (32, 32),
        "in_channels": 1,
        "out_channels": 1
    }


# -------------------- U-Net -------------------- #

def test_unet(model_manager, data_info):
    model = model_manager("unet", data_info)

    assert isinstance(model, torch.nn.Module)
    assert isinstance(model, UNet)


# -------------------- Attention U-Net -------------------- #

def test_attention_unet(model_manager, data_info):
    model = model_manager("attention unet", data_info)

    assert isinstance(model, torch.nn.Module)
    assert isinstance(model, AttentionUNet)


# -------------------- Transformer -------------------- #

def test_transformer(model_manager, data_info):
    model = model_manager("transformer", data_info)

    assert isinstance(model, torch.nn.Module)
    assert isinstance(model, Transformer)


# -------------------- SWin Transformer -------------------- #

def test_swin_transformer(model_manager, data_info):
    model = model_manager("swin transformer", data_info)

    assert isinstance(model, torch.nn.Module)
    assert isinstance(model, SWinTransformer)
