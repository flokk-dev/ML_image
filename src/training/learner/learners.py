"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: deep learning
import torch

# IMPORT: project
from src.training.learner.learner import Learner

from model import *
from loss import *
from metric import *


class UnsupervisedLearner(Learner):
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, params, data_info, weights_path):
        # Mother class
        super(UnsupervisedLearner, self).__init__(params, data_info, weights_path)

    def _learn(self, inputs, learn=True):
        raise NotImplementedError()

    def _compute_metrics(self, prediction, target):
        return {
            self._metrics[metric_name](prediction).item()
            for metric_name in self._metrics
        }

    def __call__(self, inputs, learn=True):
        raise NotImplementedError()


class SupervisedLearner(Learner):
    def __init__(self, params, data_info, weights_path):
        # Mother class
        super(SupervisedLearner, self).__init__(params, data_info, weights_path)

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
