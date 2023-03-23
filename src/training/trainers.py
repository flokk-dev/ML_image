"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *

# IMPORT: deep learning
import torch

# IMPORT: project
from src.training.trainer import Trainer

from src.loading.data_loader import UnsupervisedDataLoader, SupervisedDataLoader
from src.loading import UnsupervisedLoader, SupervisedLoader
from .components import TrainingComponentsManager


class UnsupervisedTrainer(Trainer):
    """
    Represents a trainer for unsupervised deep learning problem.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour
        _data_loaders : Dict[str: UnsupervisedDataLoader]
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

    def __init__(self, params: Dict[str, Any]):
        """
        Instantiates a UnsupervisedTrainer.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour
        """
        # Mother class
        super(UnsupervisedTrainer, self).__init__(params)

        # Attributes
        self._params = params

        # Components
        self._loader: UnsupervisedLoader = UnsupervisedLoader(self._params)

    def _learn_on_batch(self, batch: torch.Tensor, learn: bool = True):
        """
        Learns using data within a batch.

        Parameters
        ----------
            batch : torch.Tensor
                batch of tensors
            learn : bool
                boolean indicating whether to train

        Raises
        ----------
            NotImplementedError
                function isn't implemented yet
        """
        # Clear GPU cache
        torch.cuda.empty_cache()

        inputs = batch.type(torch.float32).to(torch.device(self._DEVICE))

        self._components.optimizer.zero_grad()
        with torch.set_grad_enabled(learn):
            logits = self._components.model(inputs)
            if not self._params["training_purpose"] == "regression":
                logits = torch.nn.Softmax(dim=1)(logits)

            loss = self._components.loss(logits)

            if learn:
                loss.backward()
                self._components.optimizer.step()

        return loss.item(), self._compute_metrics(logits)

    def _compute_metrics(self, prediction_batch: torch.Tensor):
        """
        Computes training's metrics.

        Parameters
        ----------
            prediction_batch : torch.Tensor
                batch of predicted tensors
        """
        return {
            self._components.metrics[metric_name](prediction_batch).item()
            for metric_name in self._components.metrics
        }

    def __call__(self, dataset_path: str, weights_path: str):
        """
        Parameters
        ----------
            dataset_path : str
                path to the dataset
            weights_path : str
                path to the model's weights
        """
        super().__call__(dataset_path, weights_path)

        # Components
        self._data_loaders: Dict[str: UnsupervisedDataLoader] = UnsupervisedLoader(
            self._params
        )(dataset_path)

        self._components: TrainingComponentsManager = TrainingComponentsManager(
            self._params, self._data_loaders["train"].data_info, weights_path
        )


class SupervisedTrainer(Trainer):
    """
    Represents a trainer for supervised deep learning problem.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour
        _data_loaders : Dict[str: SupervisedDataLoader]
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

    def __init__(self, params: Dict[str, Any]):
        """
        Instantiates a SupervisedTrainer.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour
        """
        # Mother class
        super(SupervisedTrainer, self).__init__(params)

        # Attributes
        self._params = params

        # Components
        self._loader: SupervisedLoader = SupervisedLoader(self._params)

    def _learn_on_batch(self, batch: Tuple[torch.Tensor, torch.Tensor], learn: bool = True):
        """
        Learns using data within a batch.

        Parameters
        ----------
            batch : Tuple[torch.Tensor, torch.Tensor]
                batch of tensors
            learn : bool
                boolean indicating whether to train
        """
        # Clear GPU cache
        torch.cuda.empty_cache()

        inputs = batch[0].type(torch.float32).to(torch.device(self._DEVICE))
        targets = batch[1].type(torch.float32).to(torch.device(self._DEVICE))

        self._components.optimizer.zero_grad()
        with torch.set_grad_enabled(learn):
            logits = self._components.model(inputs)
            if not self._params["training_purpose"] == "regression":
                logits = torch.nn.Softmax(dim=1)(logits)

            loss = self._components.loss(logits, targets)

            if learn:
                loss.backward()
                self._components.optimizer.step()

        return loss.item(), self._compute_metrics(logits, targets)

    def _compute_metrics(self, prediction_batch: torch.Tensor, target_batch: torch.Tensor):
        """
        Computes training's metrics.

        Parameters
        ----------
            prediction_batch : torch.Tensor
                batch of predicted tensors
            target_batch : torch.Tensor
                batch of target tensors
        """
        return {
            self._components.metrics[metric_name](prediction_batch, target_batch).item()
            for metric_name in self._components.metrics
        }

    def __call__(self, dataset_path: str, weights_path: str):
        """
        Parameters
        ----------
            dataset_path : str
                path to the dataset
            weights_path : str
                path to the model's weights
        """
        super().__call__(dataset_path, weights_path)

        # Components
        self._data_loaders: Dict[str: SupervisedDataLoader] = SupervisedLoader(
            self._params
        )(dataset_path)

        self._components: TrainingComponentsManager = TrainingComponentsManager(
            self._params, self._data_loaders["train"].data_info, weights_path
        )
