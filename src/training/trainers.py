"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: deep learning
import torch

# IMPORT: project
from src.training.trainer import Trainer
from .components import TrainingComponents

from src.loading import UnsupervisedLoader, SupervisedLoader


class UnsupervisedTrainer(Trainer):
    def __init__(self, params, dataset_path, weights_path):
        # Mother class
        super(UnsupervisedTrainer, self).__init__(params)

        # Components
        self._data_loaders = UnsupervisedLoader(self._params)(dataset_path)

        self._components = TrainingComponents(
            self._params, self._data_loaders["train"].data_info, weights_path
        )

    def _learn_on_batch(self, batch, learn=True):
        raise NotImplementedError()

    def _compute_metrics(self, prediction):
        return {
            self._components.metrics[metric_name](prediction).item()
            for metric_name in self._components.metrics
        }


class SupervisedTrainer(Trainer):
    def __init__(self, params, dataset_path, weights_path):
        # Mother class
        super(SupervisedTrainer, self).__init__(params)

        # Components
        self._data_loaders = SupervisedLoader(self._params)(dataset_path)

        self._components = TrainingComponents(
            self._params, self._data_loaders["train"].data_info, weights_path
        )

    def _learn_on_batch(self, batch, learn=True):
        # Clear GPU cache
        torch.cuda.empty_cache()

        inputs = batch[0].type(torch.float32).to(torch.device(self._DEVICE))
        targets = batch[1].type(torch.float32).to(torch.device(self._DEVICE))

        self._components.optimizer.zero_grad()
        with torch.set_grad_enabled(learn):
            logits = self._components.model(inputs)
            loss = self._components.loss(logits, targets)

            if learn:
                loss.backward()
                self._components.optimizer.step()

        return loss.item(), self._compute_metrics(logits, targets)

    def _compute_metrics(self, prediction, target):
        return {
            self._components.metrics[metric_name](prediction, target).item()
            for metric_name in self._components.metrics
        }
