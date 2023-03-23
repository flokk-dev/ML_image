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
def loss():
    return Loss()


@pytest.fixture(scope="function")
def loss_manager():
    return LossManager()


# -------------------- METRIC -------------------- #

def test_loss(loss):
    prediction = torch.Tensor().type(torch.float32)
    target = torch.Tensor().type(torch.float32)

    with pytest.raises(NotImplementedError):
        loss(prediction, target)


# -------------------- REGRESSION METRIC -------------------- #

def test_loss_regression(loss_manager):
    # METRIC
    loss = loss_manager("regression", "mean absolute error")

    # TENSORS
    prediction = torch.Tensor([[[0, 0, 1, 1]]]).type(torch.float32)  # (1, 1, 4)
    target = torch.Tensor([[[0, 0, 0, 1]]]).type(torch.float32)  # (1, 1, 4)

    loss_value = loss(prediction, target)
    assert isinstance(loss_value, torch.Tensor)


def test_loss_regressions(loss_manager):
    # TENSORS
    prediction = torch.Tensor([[[0, 0, 1, 1]]]).type(torch.float32)  # (1, 1, 4)
    target = torch.Tensor([[[0, 0, 0, 1]]]).type(torch.float32)  # (1, 1, 4)

    # METRIC
    for loss_id, loss in loss_manager["regression"].items():
        loss_value = loss()(prediction, target)
        assert isinstance(loss_value, torch.Tensor)


# -------------------- CLASSIFICATION METRIC -------------------- #

def test_loss_classification(loss_manager):
    # METRIC
    loss = loss_manager("classification", "dice")

    # TENSORS
    prediction = torch.Tensor([[[0, 0, 1, 1], [1, 1, 0, 0]]]).type(torch.float32)  # (1, 2, 4)
    target = torch.Tensor([[[0, 0, 0, 1]]]).type(torch.float32)  # (1, 1, 4)

    metric_value = loss(prediction, target)
    assert isinstance(metric_value, torch.Tensor)


def test_loss_classifications(loss_manager):
    # TENSORS
    prediction = torch.Tensor([[[0, 0, 1, 1], [1, 1, 0, 0]]]).type(torch.float32)  # (1, 2, 4)
    target = torch.Tensor([[[0, 0, 0, 1]]]).type(torch.float32)  # (1, 1, 4)

    # METRIC
    for loss_id, loss in loss_manager["classification"].items():
        loss_value = loss()(prediction, target)
        assert isinstance(loss_value, torch.Tensor)
