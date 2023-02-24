"""
Creator: HOCQUET Florian, LANDZA Houdi
Date: 30/09/2022
Version: 1.0

Purpose: Links the training's data to a web visualiser.
"""

# IMPORT: utils
import os

# IMPORT: data visualization
import wandb

# IMPORT: deep learning
import torch


class Visualizer:
    """
    The class plotting training's results.

    Methods
    - collect_info: Collect information within the model.
    - plot_res: Plots the losses on WandB website.
    - plot_images: Plots the images on WandB website.
    - _get_images: Gets images from a list for the plotting of the results.
    - finish_run: Ends the WandB process.
    """

    def __init__(self, model_name: str, params: dict, mode="online"):
        """
        Initialize the Visualizer class.

        Parameters:
        - model_name (str): the name of the trained model.
        - params (dict): the training's parameters.
        - entity (str): the WandB entity to use.
        - project (str): the WandB project to use.
        """
        # Initialize the wandb entity
        os.environ["WANDB_SILENT"] = "true"
        wandb.init(entity="pic_chb_seg_aorte", project="revue", mode=mode)

        # Set the run's name.
        wandb.run.name = model_name

        # Configure wandb's parameters
        wandb.config = params

    @staticmethod
    def collect_info(model):
        """
        Collect information within the model.

        Parameters:
        - model (): the trained model.
        """
        wandb.watch(model)

    @staticmethod
    def plot_res(loss: dict, metrics: dict, epoch: int, lr: list):
        """
        Plots the losses on WandB website.

        Parameters:
        - losses (list): the loss obtained during the training.
        - metrics (list): the metrics obtained during the training.
        - epoch (int): the current epoch of the training.
        - lr (list): the current epoch's learning rate.
        """
        result = dict()

        # METRICS
        for key in metrics.keys():
            result[f"train {key}"] = metrics[key]["train"][epoch]
            result[f"valid {key}"] = metrics[key]["valid"][epoch]

        # LOSSES
        result["train loss"] = loss["train"][epoch]
        result["valid loss"] = loss["valid"][epoch]

        # LEARNING RATE
        result["lr"] = lr

        # LOG ON WANDB
        wandb.log(result)

    @staticmethod
    def plot_images(inputs: torch.Tensor, targets: torch.Tensor, preds: torch.Tensor, step: str):
        """
        Plots the images on WandB website.

        Parameters:
        - inputs (str): the input tensors.
        - targets (str): the target tensors.
        - targets (str): the predicted tensors.
        - step (str): the current training's step.
        """
        raise NotImplementedError()

    @staticmethod
    def finish_run():
        """
        Ends the WandB process.
        """
        wandb.finish()
