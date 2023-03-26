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


# -------------------- METRIC -------------------- #

def test_metric(params_binary_class):
    prediction = torch.Tensor().type(torch.float32)
    target = torch.Tensor().type(torch.float32)

    with pytest.raises(NotImplementedError):
        Metric(params_binary_class)(prediction, target)


# -------------------- REGRESSION METRIC -------------------- #

def test_metric_regression(params_binary_class):
    # METRIC
    metric = MetricManager(params_binary_class)("regression", "mean absolute error")

    # TENSORS
    prediction = torch.Tensor([[[0, 0, 1, 1]]]).type(torch.float32)  # (1, 1, 4)
    target = torch.Tensor([[[0, 0, 0, 1]]]).type(torch.float32)  # (1, 1, 4)

    metric_value = metric(prediction, target)
    assert isinstance(metric_value, torch.Tensor)


def test_metric_regressions(params_binary_class):
    # TENSORS
    prediction = torch.Tensor([[[0, 0, 1, 1]]]).type(torch.float32)  # (1, 1, 4)
    target = torch.Tensor([[[0, 0, 0, 1]]]).type(torch.float32)  # (1, 1, 4)

    # METRIC
    for metric_id, metric in MetricManager(params_binary_class)["regression"].items():
        metric_value = metric(params_binary_class)(prediction, target)
        assert isinstance(metric_value, torch.Tensor)


# -------------------- CLASSIFICATION METRIC -------------------- #

def test_metric_classification_binary_class(params_binary_class):
    # METRIC
    metric = MetricManager(params_binary_class)("classification", "accuracy")

    # TENSORS
    prediction = torch.Tensor([[[0, 0, 0, 1]]]).type(torch.float32)  # (1, 1, 4)
    target = torch.Tensor([[[0, 0, 0, 1]]]).type(torch.float32)  # (1, 1, 4)

    metric_value = metric(prediction, target)
    assert isinstance(metric_value, torch.Tensor)


def test_metric_classifications_binary_class(params_binary_class):
    # TENSORS
    prediction = torch.Tensor([[[0, 0, 0, 1]]]).type(torch.float32)  # (1, 1, 4)
    target = torch.Tensor([[[0, 0, 0, 1]]]).type(torch.float32)  # (1, 1, 4)

    # METRIC
    for metric_id, metric in MetricManager(params_binary_class)["classification"].items():
        metric_value = metric(params_binary_class)(prediction, target)
        assert isinstance(metric_value, torch.Tensor)


def test_metric_classification_multi_class(params_multi_class):
    # METRIC
    metric = MetricManager(params_multi_class)("classification", "accuracy")

    # TENSORS
    prediction = torch.Tensor([[[0, 0, 1, 1], [1, 1, 0, 0]]]).type(torch.float32)  # (1, 2, 4)
    target = torch.Tensor([[[0, 0, 0, 1]]]).type(torch.float32)  # (1, 1, 4)

    metric_value = metric(prediction, target)
    assert isinstance(metric_value, torch.Tensor)


def test_metric_classifications_multi_class(params_multi_class):
    # TENSORS
    prediction = torch.Tensor([[[0, 0, 1, 1], [1, 1, 0, 0]]]).type(torch.float32)  # (1, 2, 4)
    target = torch.Tensor([[[0, 0, 0, 1]]]).type(torch.float32)  # (1, 1, 4)

    # METRIC
    for metric_id, metric in MetricManager(params_multi_class)["classification"].items():
        metric_value = metric(params_multi_class)(prediction, target)
        assert isinstance(metric_value, torch.Tensor)
