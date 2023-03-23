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
from src.training.components.model.unet import UNet
from src.training.components.early_stopper import EarlyStopper


# -------------------- FIXTURES -------------------- #

@pytest.fixture(scope="function")
def early_stopper():
    return EarlyStopper({"duration": 5}, loss_behaviour="minimization")


@pytest.fixture(scope="function")
def early_stopper_minimization():
    return EarlyStopper({"duration": 5}, loss_behaviour="minimization")


@pytest.fixture(scope="function")
def early_stopper_maximization():
    return EarlyStopper({"duration": 5}, loss_behaviour="maximization")


# -------------------- EARLY STOPPER -------------------- #

def test_early_stopper(early_stopper):
    assert early_stopper._max_duration == 5

    assert early_stopper["epoch"] is None
    assert early_stopper["duration"] == 0
    assert early_stopper["weights"] is None


# -------------------- EARLY STOPPER MINIMIZATION -------------------- #

def test_early_stopper_minimization(early_stopper_minimization):
    assert early_stopper_minimization["loss_value"] == float("inf")

    assert early_stopper_minimization._evolves_well(0.5, 0.75)
    assert not early_stopper_minimization._evolves_well(0.75, 0.5)


def test_early_stopper_minimization_check_epoch(early_stopper_minimization):
    # RESULTS
    early_stopper_epochs = [
        {"loss": 0.5, "epoch": 0, "duration": 0, "return": True},
        {"loss": 0.4, "epoch": 1, "duration": 0, "return": True},
        {"loss": 0.3, "epoch": 2, "duration": 0, "return": True},
        {"loss": 0.4, "epoch": 2, "duration": 1, "return": True},
        {"loss": 0.5, "epoch": 2, "duration": 2, "return": True},
        {"loss": 0.6, "epoch": 2, "duration": 3, "return": True},
        {"loss": 0.7, "epoch": 2, "duration": 4, "return": True},
        {"loss": 0.8, "epoch": 2, "duration": 5, "return": False}
    ]

    # MODEL
    model = torch.nn.DataParallel(
        UNet(data_info={
            "spatial_dims": 2, "img_size": (64, 64),
            "in_channels": 1, "out_channels": 1
        })
    )

    for idx, epoch_log in enumerate(early_stopper_epochs):
        result = early_stopper_minimization.check_epoch(
            idx, epoch_log["loss"],
            model.module.state_dict()
        )

        assert early_stopper_minimization["epoch"] == epoch_log["epoch"]
        assert early_stopper_minimization["duration"] == epoch_log["duration"]
        assert result == epoch_log["return"]


# -------------------- EARLY STOPPER MAXIMIZATION -------------------- #

def test_early_stopper_maximization(early_stopper_maximization):
    assert early_stopper_maximization["loss_value"] == float("-inf")

    assert early_stopper_maximization._evolves_well(0.75, 0.5)
    assert not early_stopper_maximization._evolves_well(0.5, 0.75)


def test_early_stopper_maximization_check_epoch(early_stopper_maximization):
    # RESULTS
    early_stopper_epochs = [
        {"loss": 0.5, "epoch": 0, "duration": 0, "return": True},
        {"loss": 0.6, "epoch": 1, "duration": 0, "return": True},
        {"loss": 0.7, "epoch": 2, "duration": 0, "return": True},
        {"loss": 0.6, "epoch": 2, "duration": 1, "return": True},
        {"loss": 0.5, "epoch": 2, "duration": 2, "return": True},
        {"loss": 0.4, "epoch": 2, "duration": 3, "return": True},
        {"loss": 0.3, "epoch": 2, "duration": 4, "return": True},
        {"loss": 0.2, "epoch": 2, "duration": 5, "return": False}
    ]

    # MODEL
    model = torch.nn.DataParallel(
        UNet(data_info={
            "spatial_dims": 2, "img_size": (64, 64),
            "in_channels": 1, "out_channels": 1
        })
    )

    for idx, epoch_log in enumerate(early_stopper_epochs):
        result = early_stopper_maximization.check_epoch(
            idx, epoch_log["loss"],
            model.module.state_dict()
        )

        assert early_stopper_maximization["epoch"] == epoch_log["epoch"]
        assert early_stopper_maximization["duration"] == epoch_log["duration"]
        assert result == epoch_log["return"]
