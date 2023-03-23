"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *

import time
from tqdm import tqdm

# IMPORT: deep learning
import torch

# IMPORT: project
from src.loading import Loader
from src.loading.data_loader import DataLoader
from .components import TrainingComponentsManager


class Trainer:
    """
    Represents a general Trainer, that will be derived depending on the use case.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour
        _data_loaders : Dict[str: DataLoader]
            training and validation data loaders
        _components : TrainingComponentsManager
            training components

    Methods
    ----------
        _launch
            Launches the training
        _run_epoch
            Runs an epoch
        _learn_on_batch
            Learns using data within a batch
        _compute_metrics
            Computes training's metrics
    """

    _DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def __init__(self, params: Dict[str, Any]):
        """
        Instantiates a Trainer.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour
        """
        # Attributes
        self._params: Dict[str, Any] = params

        # Components
        self._loader: Loader = None
        self._data_loaders: Dict[str: DataLoader] = None

        self._components: TrainingComponentsManager = None

    def _launch(self):
        """ Launches the training. """
        print("\nLancement de l'entrainement.")

        time.sleep(1)
        for epoch in tqdm(self._params["nb_epochs"]):
            # Clear cache
            torch.cuda.empty_cache()

            # Learn
            self._run_epoch(step="train")
            self._run_epoch(step="valid")

            # Update the epoch
            self._components.dashboard.upload_values(self._components.scheduler.get_last_lr()[0])
            self._components.scheduler.step()

        # End the training
        time.sleep(30)
        self._components.dashboard.shutdown()

    def _run_epoch(self, step: str):
        """
        Runs an epoch.

        Parameters
        ----------
            step : str
                training step
        """
        epoch_loss = list()
        epoch_metrics = {metric_name: list() for metric_name in self._params["metrics"]}

        batch_idx = 0
        learning_allowed = step == "train"

        for sub_data_loader in self._data_loaders[step]:
            for batch in sub_data_loader:
                batch_loss, batch_metrics = self._learn_on_batch(batch, learn=learning_allowed)

                epoch_loss.append(batch_loss)
                for metric_name in epoch_metrics.keys():
                    epoch_metrics[metric_name].append(batch_metrics[metric_name])

                batch_idx += 1

        # Store the results
        self._components.dashboard.update_loss_metrics(epoch_loss, epoch_metrics, step)

    def _learn_on_batch(
            self,
            batch: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], learn: bool = True
    ):
        """
        Learns using data within a batch.

        Parameters
        ----------
            batch : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
                batch of tensors
            learn : bool
                boolean indicating whether to train
        """
        raise NotImplementedError()

    def _compute_metrics(self, prediction_batch: torch.Tensor):
        """
        Computes training's metrics.

        Parameters
        ----------
            prediction_batch : torch.Tensor
                batch of predicted tensors

        Raises
        ----------
            NotImplementedError
                function isn't implemented yet
        """
        raise NotImplementedError()

    def __call__(self, dataset_path: str, weights_path: str):
        """
        Parameters
        ----------
            dataset_path : str
                path to the dataset
            weights_path : str
                path to the model's weights

        Raises
        ----------
            NotImplementedError
                function isn't implemented yet
        """
        raise NotImplementedError()
