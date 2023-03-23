"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *
import torch

# IMPORT: data visualization
import wandb

# IMPORT: project
from .dashboard import Dashboard


class Dashboard2D(Dashboard):
    """
    Represents a dashboard for 2D data.

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
        Instantiates a Dashboard2D.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour
            train_id : str
                id of the training
        """
        # Mother Class
        super(Dashboard2D, self).__init__(params, train_id, mode=mode)

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
        """
        images = {
            f"input_{step}": input_batch[0],
            f"target_{step}": target_batch[0],
            f"prediction_{step}": prediction_batch[0],
        }

        for title, image in images.items():
            images[title] = [wandb.Image(image.data)]

        wandb.log(images)


class Dashboard25D(Dashboard):
    """
    Represents a dashboard for 2D data.

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
        Instantiates a Dashboard25D.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour
            train_id : str
                id of the training
        """
        # Mother Class
        super(Dashboard25D, self).__init__(params, train_id, mode=mode)

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
        """
        images = {
            f"input_{step}": input_batch[0, input_batch.shape[1] // 2],
            f"target_{step}": target_batch[0][0],
            f"prediction_{step}": prediction_batch[0][0],
        }

        for title, image in images.items():
            images[title] = [wandb.Image(image.data)]

        wandb.log(images)


class Dashboard3D(Dashboard):
    """
    Represents a dashboard for 2D data.

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
        Instantiates a Dashboard3D.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour
            train_id : str
                id of the training
        """
        # Mother Class
        super(Dashboard3D, self).__init__(params, train_id, mode=mode)

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
        """
        prediction_batch = (prediction_batch[0, 1] >= 0.50).float()

        images = {
            f"input_{step}": input_batch[0, 0, input_batch.shape[2] // 2],
            f"target_{step}": target_batch[0][0][0],
            f"prediction_{step}": prediction_batch[0][0][0],
        }

        for title, image in images.items():
            images[title] = [wandb.Image(image.data)]

        wandb.log(images)
