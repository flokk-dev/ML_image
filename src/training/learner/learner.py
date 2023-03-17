"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import sys

# IMPORT: deep learning
import torch

# IMPORT: project
import utils
from src.training.early_stopper import EarlyStopper

from model import *
from loss import *
from metric import *


class Learner:
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, params, weights_path):
        # Attributes
        self._params = params

        # Model
        self._model = utils.str_to_class(self._params["model"])(
            (32, 5, 64, 64), (32, 5, 64, 64), weights_path=weights_path
        )
        self._model = torch.nn.DataParallel(self._model)

        # Optimizer
        self._optimizer = torch.optim.Adam(params=self._model.parameters(), lr=self._params["lr"])
        self._scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            self._optimizer, self._params["lr_multiplier"]
        )

        # Loss
        self._loss = utils.str_to_class(params["loss"])

        # Metrics
        self._metrics = {
            metric_name: utils.str_to_class(metric_name)()
            for metric_name in self._params["metrics"]
        }

    def _learn(self, inputs, targets, learn=True):
        # Clear GPU cache
        torch.cuda.empty_cache()

        inputs = inputs.type(torch.float32).to(torch.device(self._DEVICE))
        targets = targets.type(torch.float32).to(torch.device(self._DEVICE))

        self._optimizer.zero_grad()
        with torch.set_grad_enabled(learn):
            logits = self._model(inputs)
            loss = self._loss(logits, targets)

            if learn:
                loss.backward()
                self._optimizer.step()

        return loss.item(), self._compute_metrics(logits, targets)

    def _compute_metrics(self, prediction, target):
        return {
            self._metrics[metric_name](prediction, target).item()
            for metric_name in self._metrics
        }

    def __call__(self, inputs, targets, learn=True):
        self._learn(inputs, targets, learn)
