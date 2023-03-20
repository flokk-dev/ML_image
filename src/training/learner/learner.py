"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: deep learning
import torch

# IMPORT: project
import utils

from .model import *
from .loss import *
from .metric import *


class Learner:
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, params, data_info, weights_path):
        # Attributes
        self._params = params

        # Model
        self._model = utils.str_to_class(self._params["model"])(
            data_info, weights_path=weights_path
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

    @property
    def scheduler(self):
        return self._scheduler

    def _learn(self, inputs, learn=True):
        raise NotImplementedError()

    def _compute_metrics(self, prediction, target):
        raise NotImplementedError()

    def __call__(self, inputs, learn=True):
        self._learn(inputs, learn)
