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
from src.training.components.loss.loss import Loss
from src.training.components.loss import LossManager


# -------------------- FIXTURES -------------------- #

@pytest.fixture(scope="function")
def params_binary_class():
    return {
        "spatial_dims": 2, "img_size": (64, 64),
        "in_channels": 1, "out_channels": 1
    }


@pytest.fixture(scope="function")
def params_multi_class():
    return {
        "spatial_dims": 2, "img_size": (64, 64),
        "in_channels": 1, "out_channels": 2
    }


# -------------------- loss -------------------- #

def test_loss(params_binary_class):
    prediction = torch.Tensor().type(torch.float32)
    target = torch.Tensor().type(torch.float32)

    with pytest.raises(NotImplementedError):
        Loss(params_binary_class)(prediction, target)


# -------------------- REGRESSION loss -------------------- #

def test_loss_regression(params_binary_class):
    # LOSS
    loss = LossManager(params_binary_class)("regression", "mean absolute error")

    # TENSORS
    prediction = torch.Tensor([[[0, 0, 1, 1]]]).type(torch.float32)  # (1, 1, 4)
    target = torch.Tensor([[[0, 0, 0, 1]]]).type(torch.float32)  # (1, 1, 4)

    loss_value = loss(prediction, target)
    assert isinstance(loss_value, torch.Tensor)


def test_loss_regressions(params_binary_class):
    # TENSORS
    prediction = torch.Tensor([[[0, 0, 1, 1]]]).type(torch.float32)  # (1, 1, 4)
    target = torch.Tensor([[[0, 0, 0, 1]]]).type(torch.float32)  # (1, 1, 4)

    # LOSS
    for loss_id, loss in LossManager(params_binary_class)["regression"].items():
        loss_value = loss(params_binary_class)(prediction, target)
        assert isinstance(loss_value, torch.Tensor)


# -------------------- CLASSIFICATION loss -------------------- #

def test_loss_classification_binary_class(params_binary_class):
    # LOSS
    loss = LossManager(params_binary_class)("classification", "dice")

    # TENSORS
    prediction = torch.Tensor([[[0, 0, 0, 1]]]).type(torch.float32)  # (1, 1, 4)
    target = torch.Tensor([[[0, 0, 0, 1]]]).type(torch.float32)  # (1, 1, 4)

    loss_value = loss(prediction, target)
    assert isinstance(loss_value, torch.Tensor)


def test_loss_classifications_binary_class(params_binary_class):
    # TENSORS
    prediction = torch.Tensor([[[0, 0, 0, 1]]]).type(torch.float32)  # (1, 1, 4)
    target = torch.Tensor([[[0, 0, 0, 1]]]).type(torch.float32)  # (1, 1, 4)

    # LOSS
    for loss_id, loss in LossManager(params_binary_class)["classification"].items():
        loss_value = loss(params_binary_class)(prediction, target)
        assert isinstance(loss_value, torch.Tensor)


def test_loss_classification_multi_class(params_multi_class):
    # LOSS
    loss = LossManager(params_multi_class)("classification", "dice")

    # TENSORS
    prediction = torch.Tensor([[[0, 0, 1, 1], [1, 1, 0, 0]]]).type(torch.float32)  # (1, 2, 4)
    target = torch.Tensor([[[0, 0, 0, 1]]]).type(torch.float32)  # (1, 1, 4)

    loss_value = loss(prediction, target)
    assert isinstance(loss_value, torch.Tensor)


def test_loss_classifications_multi_class(params_multi_class):
    # TENSORS
    prediction = torch.Tensor([[[0, 0, 1, 1], [1, 1, 0, 0]]]).type(torch.float32)  # (1, 2, 4)
    target = torch.Tensor([[[0, 0, 0, 1]]]).type(torch.float32)  # (1, 1, 4)

    # LOSS
    for loss_id, loss in LossManager(params_multi_class)["classification"].items():
        loss_value = loss(params_multi_class)(prediction, target)
        assert isinstance(loss_value, torch.Tensor)
