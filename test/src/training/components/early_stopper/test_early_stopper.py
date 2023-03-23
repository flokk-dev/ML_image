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
    return EarlyStopper({"duration": 5})


# -------------------- EARLY STOPPER -------------------- #

def test_early_stopper(early_stopper):
    assert early_stopper["loss_value"] == float("inf")
    assert early_stopper["epoch"] is None
    assert early_stopper["duration"] == 0
    assert early_stopper["weights"] is None


def test_early_stopper_check_epoch(early_stopper):
    # RESULTS
    early_stopper_epochs = [
        {"loss": 0.9, "epoch": 0, "duration": 0, "return": True},
        {"loss": 0.89, "epoch": 1, "duration": 0, "return": True},
        {"loss": 0.88, "epoch": 2, "duration": 0, "return": True},
        {"loss": 0.89, "epoch": 2, "duration": 1, "return": True},
        {"loss": 0.90, "epoch": 2, "duration": 2, "return": True},
        {"loss": 0.91, "epoch": 2, "duration": 3, "return": True},
        {"loss": 0.92, "epoch": 2, "duration": 4, "return": True},
        {"loss": 0.93, "epoch": 2, "duration": 5, "return": False}
    ]

    # MODEL
    model = torch.nn.DataParallel(
        UNet(data_info={
            "spatial_dims": 2, "img_size": (64, 64),
            "in_channels": 1, "out_channels": 1
        })
    )

    for idx, epoch_log in enumerate(early_stopper_epochs):
        result = early_stopper.check_epoch(idx, epoch_log["loss"], model)

        assert early_stopper["epoch"] == epoch_log["epoch"]
        assert early_stopper["duration"] == epoch_log["duration"]
        assert result == epoch_log["return"]
