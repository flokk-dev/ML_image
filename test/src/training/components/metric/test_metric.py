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
from src.training.components.metric.metric import Metric
from src.training.components.metric import MetricManager


# -------------------- FIXTURES -------------------- #

@pytest.fixture(scope="function")
def metric():
    return Metric()


@pytest.fixture(scope="function")
def metric_manager():
    return MetricManager()


# -------------------- METRIC -------------------- #

def test_metric(metric):
    prediction = torch.Tensor().type(torch.float32)
    target = torch.Tensor().type(torch.float32)

    with pytest.raises(NotImplementedError):
        metric(prediction, target)


# -------------------- REGRESSION METRIC -------------------- #

def test_metric_regression(metric_manager):
    # METRIC
    metric = metric_manager("regression", "mean absolute error")

    # TENSORS
    prediction = torch.Tensor([[[0, 0, 1, 1]]]).type(torch.float32)  # (1, 1, 4)
    target = torch.Tensor([[[0, 0, 0, 1]]]).type(torch.float32)  # (1, 1, 4)

    metric_value = metric(prediction, target)
    assert isinstance(metric_value, torch.Tensor)


def test_metric_regressions(metric_manager):
    # TENSORS
    prediction = torch.Tensor([[[0, 0, 1, 1]]]).type(torch.float32)  # (1, 1, 4)
    target = torch.Tensor([[[0, 0, 0, 1]]]).type(torch.float32)  # (1, 1, 4)

    # METRIC
    for metric_id, metric in metric_manager["regression"].items():
        metric_value = metric()(prediction, target)
        assert isinstance(metric_value, torch.Tensor)


# -------------------- CLASSIFICATION METRIC -------------------- #

def test_metric_classification(metric_manager):
    # METRIC
    metric = metric_manager("classification", "accuracy")

    # TENSORS
    prediction = torch.Tensor([[[0, 0, 1, 1], [1, 1, 0, 0]]]).type(torch.float32)  # (1, 2, 4)
    target = torch.Tensor([[[0, 0, 0, 1]]]).type(torch.float32)  # (1, 1, 4)

    metric_value = metric(prediction, target)
    assert isinstance(metric_value, torch.Tensor)


def test_metric_classifications(metric_manager):
    # TENSORS
    prediction = torch.Tensor([[[0, 0, 1, 1], [1, 1, 0, 0]]]).type(torch.float32)  # (1, 2, 4)
    target = torch.Tensor([[[0, 0, 0, 1]]]).type(torch.float32)  # (1, 1, 4)

    # METRIC
    for metric_id, metric in metric_manager["classification"].items():
        metric_value = metric()(prediction, target)
        assert isinstance(metric_value, torch.Tensor)
