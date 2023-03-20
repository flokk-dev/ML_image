"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: test
import pytest

# IMPORT: project
from src.training.early_stopper import EarlyStopper
from src.training.learner.model import UNet


# -------------------- FIXTURES -------------------- #

@pytest.fixture(scope="function")
def early_stopper():
    return EarlyStopper({"duration": 5})


# -------------------- EARLY STOPPER -------------------- #

def test_early_stopper(early_stopper):
    loss_values = [1., 0.8, 0.5, 0.6, 0.3]
    for loss_value in loss_values:
        pass
