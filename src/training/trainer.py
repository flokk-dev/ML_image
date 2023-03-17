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
from src.loading import UnsupervisedLoader, SupervisedLoader
from .early_stopper import EarlyStopper
from .learner import Learner


class Trainer:
    _LOADERS = {"unsupervised": UnsupervisedLoader, "supervised": SupervisedLoader}

    def __init__(self, params, weights_path):
        # Attributes
        self._params = params
        self._name = None

        self._loss_res = {"train": list(), "valid": list()}
        self._metrics_res = {
            metric_name: {"train": list(), "valid": list()}
            for metric_name in self._params["metrics"]
        }

        # Components
        self._loader = self._LOADERS[self._params["training_type"]](self._params)
        self._learner = Learner(self._params, weights_path)

        self._early_stopper = EarlyStopper(self._params)
        self._dashboard = None

    def _train(self):
        print("\nLancement de l'entrainement.")

        time.sleep(1)
        for epoch_idx in tqdm(range(self._params["epochs"])):
            # Clear cache
            torch.cuda.empty_cache()

            # Learn
            self._train_epoch(step="train")
            self._train_epoch(step="valid")

            # Update the dashboard
            # self._dashboard.update()

            # self._scheduler.step()

        # Save the results and end the visualization
        # self._save()
        time.sleep(60)

        # self._dashboard.shutdown()
    def _train_epoch(self, step):
        epoch_loss = list()
        epoch_metric = dict()
        for key in self._metrics:
            epoch_metric[key] = list()
            epoch_metric[f"{key}_inputs"] = list()

        for batch in self._loaders[step]:
            for i in range(0, batch[0].shape[0], self._params["batch_size"]):
                batch_loss, batch_metric = self._train_block(
                    inputs=batch[0][i: i + self._params["batch_size"]],
                    targets=batch[1][i: i + self._params["batch_size"]],
                    batch_idx=batch_idx,
                    step=step
                )

                epoch_loss.append(batch_loss)
                for key in epoch_metric.keys():
                    epoch_metric[key].append(batch_metric)

                batch_idx += 1

        # Results
        current_loss = sum(epoch_loss) / len(epoch_loss)
        self._loss_results[step].append(current_loss)

        for key in epoch_metric.keys():
            if key not in self._metrics_results:
                self._metrics_results[key] = {"train": list(), "valid": list()}
            self._metrics_results[key][step].append(
                sum(epoch_metric[key]) / len(epoch_metric[key]))

    def __call__(self):
        # Clean cache
        torch.cuda.empty_cache()

        self._train()
