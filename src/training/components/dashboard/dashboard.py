"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *
import os

import torch

# IMPORT: visualization
import wandb


class Dashboard:
    """
    Represents a general dashboard, that will be derived depending on the use case.

    Attributes
    ----------
        _loss : Dict[str, List[float]]
            history of the loss value during training
        _metrics : Dict[str, Dict[str, List[float]]]
            history of the metrics values during training

    Methods
    ----------
        collect_info
            Specifies the model to follow
        shutdown
            Shutdowns the dashboard
        update_loss_metrics
            Updates the history of loss and metrics
        upload_values
            Uploads the history of learning rate, loss and metrics
        upload_images
            Uploads examples of results
    """

    def __init__(self, params: Dict[str, Any], train_id: str, mode="online"):
        """
        Instantiates a Dashboard.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour
            train_id : str
                id of the training
        """
        # Initialize the wandb entity
        os.environ["WANDB_SILENT"] = "true"
        wandb.init(entity="machine_learning_lib", project="test", mode=mode)

        wandb.run.name = train_id
        wandb.config = params

        # Attributes
        self._loss: Dict[str, List[float]] = {"train": list(), "valid": list()}

        self._metrics: Dict[str, Dict[str, List[float]]] = {
            metric_name: {"train": list(), "valid": list()}
            for metric_name in params["metrics"]
        }

    @staticmethod
    def collect_info(model: torch.nn.Module):
        """
        Specifies the model to follow.

        Parameters
        ----------
            model : torch.nn.Module
                model to follow.
        """
        wandb.watch(model)

    @staticmethod
    def shutdown():
        """ Shutdowns the dashboard. """
        wandb.finish()

    def update_loss_metrics(self, loss: List[float], metrics: Dict[str, List[float]], step: str):
        """
        Updates the history of loss and metrics.

        Parameters
        ----------
            loss : List[float]
                loss values during an epoch
            metrics : Dict[str, List[float]]
                metrics values during an epoch
            step : str
                training step
        """
        # Update the loss
        current_loss = sum(loss) / len(loss)
        self._loss[step].append(current_loss)

        # Update the metrics
        for metric_name, metric_values in metrics.items():
            self._metrics[metric_name][step].append(
                sum(metrics[metric_name]) / len(metrics[metric_name])
            )

    def upload_values(self, lr):
        """
        Uploads the history of learning rate, loss and metrics.

        Parameters
        ----------
            lr : float
                learning rate value during the epoch
        """
        result = dict()

        # METRICS
        for metric_name, metric_values in self._metrics.items():
            result[f"train {metric_name}"] = metric_values["train"][-1]
            result[f"valid {metric_name}"] = metric_values["valid"][-1]

        # LOSSES
        result["train loss"] = self._loss["train"][-1]
        result["valid loss"] = self._loss["valid"][-1]

        # LEARNING RATE
        result["lr"] = lr

        # LOG ON WANDB
        wandb.log(result)

    @staticmethod
    def upload_images(
            input_batch: torch.Tensor, prediction_batch: torch.Tensor, target_batch: torch.Tensor,
            step: str
    ):
        """
        Uploads examples of results.

        Parameters
        ----------
            input_batch : torch.Tensor
                batch of input tensors
            prediction_batch : torch.Tensor
                batch of predicted tensors
            target_batch : torch.Tensor
                batch of target tensors
            step : str
                training step

        Raises
        ----------
            NotImplementedError
                function isn't implemented yet
        """
        raise NotImplementedError()
