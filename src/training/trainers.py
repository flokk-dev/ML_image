"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import time
from tqdm import tqdm

# IMPORT: deep learning
import torch

# IMPORT: project
from src.training.trainer import Trainer
from src.loading import UnsupervisedLoader, SupervisedLoader
from src.training.learner import UnsupervisedLearner, SupervisedLearner


class UnsupervisedTrainer(Trainer):
    def __init__(self, params, dataset_path, weights_path):
        # Mother class
        super(UnsupervisedTrainer, self).__init__(params)

        # Components
        loader = UnsupervisedLoader(self._params)(dataset_path)
        self._data_loaders = loader(dataset_path)

        self._learner = UnsupervisedLearner(
            self._params, self._data_loaders["train"].data_info, weights_path
        )

    def _run_epoch(self, step):
        epoch_loss = list()
        epoch_metrics = {metric_name: list() for metric_name in self._params["metrics"]}

        batch_idx = 0
        learning_allowed = step == "train"

        for sub_data_loader in self._data_loaders[step]:
            for inputs in sub_data_loader:
                batch_loss, batch_metrics = self._learner(inputs, learn=learning_allowed)

                epoch_loss.append(batch_loss)
                for metric_name in epoch_metrics.keys():
                    epoch_metrics[metric_name].append(batch_metrics[metric_name])

                batch_idx += 1

        # Store the results
        self._dashboard.update_loss_metrics(epoch_loss, epoch_metrics)

    def __call__(self):
        # Clean cache
        torch.cuda.empty_cache()

        self._launch()


class SupervisedTrainer(Trainer):
    def __init__(self, params, dataset_path, weights_path):
        # Mother class
        super(SupervisedTrainer, self).__init__(params)

        # Components
        loader = SupervisedLoader(self._params)(dataset_path)
        self._data_loaders = loader(dataset_path)

        self._learner = SupervisedLearner(
            self._params, self._data_loaders["train"].data_info, weights_path
        )

    def _run_epoch(self, step):
        epoch_loss = list()
        epoch_metrics = {metric_name: list() for metric_name in self._params["metrics"]}

        batch_idx = 0
        learning_allowed = step == "train"

        for sub_data_loader in self._data_loaders[step]:
            for inputs, targets in sub_data_loader:
                batch_loss, batch_metrics = self._learner(inputs, targets, learn=learning_allowed)

                epoch_loss.append(batch_loss)
                for metric_name in epoch_metrics.keys():
                    epoch_metrics[metric_name].append(batch_metrics[metric_name])

                batch_idx += 1

        # Store the results
        self._dashboard.update_loss_metrics(epoch_loss, epoch_metrics)

    def __call__(self):
        # Clean cache
        torch.cuda.empty_cache()

        self._launch()
