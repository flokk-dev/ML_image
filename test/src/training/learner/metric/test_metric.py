"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: test
import pytest

# IMPORT: data processing
import torch

# IMPORT: project
from src.training.learner.metric.metric import Metric

from src.training.learner.metric import \
    Accuracy, JaccardIndex, F1Score, Precision, Recall, \
    MAE, MSE, RMSE, PSNR, SSIM


# -------------------- FIXTURES -------------------- #

@pytest.fixture(scope="function")
def prediction_target_tensors():
    return torch.Tensor([0, 0, 1, 1]), torch.Tensor([0, 0, 0, 1])


@pytest.fixture(scope="function")
def metric():
    return Metric()


# -------------------- METRIC -------------------- #

def test_metric(metric, prediction_target_tensors):
    with pytest.raises(NotImplementedError):
        metric(*prediction_target_tensors)


# -------------------- CLASSIFICATION METRIC -------------------- #

def test_early_stopper(early_stopper):
    loss_values = [1., 0.8, 0.5, 0.6, 0.3]
    for loss_value in loss_values:
        pass


# -------------------- REGRESSION METRIC -------------------- #
