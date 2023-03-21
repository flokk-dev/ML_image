"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import typing

import time
from tqdm import tqdm

# IMPORT: deep learning
import torch


class Trainer:
    _DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def __init__(self, params):
        # Attributes
        self._params = params
        self._name = None

        # Components
        self._data_loaders = None
        self._learner = None

        self._components = None

    def _launch(self):
        print("\nLancement de l'entrainement.")

        time.sleep(1)
        for epoch in tqdm(self._params["nb_epochs"]):
            # Clear cache
            torch.cuda.empty_cache()

            # Learn
            self._run_epoch(step="train")
            self._run_epoch(step="valid")

            # Update the epoch
            self._components.dashboard.upload_values(self._learner.scheduler.get_last_lr()[0])
            self._components.scheduler.step()

        # End the training
        time.sleep(30)
        self._components.dashboard.shutdown()

    def _run_epoch(self, step):
        epoch_loss = list()
        epoch_metrics = {metric_name: list() for metric_name in self._params["metrics"]}

        batch_idx = 0
        learning_allowed = step == "train"

        for sub_data_loader in self._data_loaders[step]:
            for batch in sub_data_loader:
                batch_loss, batch_metrics = self._learner(batch, learn=learning_allowed)

                epoch_loss.append(batch_loss)
                for metric_name in epoch_metrics.keys():
                    epoch_metrics[metric_name].append(batch_metrics[metric_name])

                batch_idx += 1

        # Store the results
        self._components.dashboard.update_loss_metrics(epoch_loss, epoch_metrics)

    def _learn_on_batch(self, inputs, learn=True):
        raise NotImplementedError()

    def _compute_metrics(self, prediction):
        raise NotImplementedError()

    def __call__(self):
        # Clean cache
        torch.cuda.empty_cache()

        # Launch training
        self._launch()
