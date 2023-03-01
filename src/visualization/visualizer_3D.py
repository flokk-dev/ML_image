"""
Creator: HOCQUET Florian, LANDZA Houdi
Date: 30/09/2022
Version: 1.0

Purpose: Links the training's data to a web visualiser.
"""

# IMPORT: data visualization
import wandb

# IMPORT: deep learning
import torch

# IMPORT: project
from .visualizer import Visualizer


class Visualizer3D(Visualizer):
    """
    The class plotting training's results.

    Methods
    - collect_info: Collect information within the models.
    - plot_res: Plots the losses on WandB website.
    - plot_images: Plots the images on WandB website.
    - _get_images: Gets images from a list for the plotting of the results.
    - finish_run: Ends the WandB process.
    """

    def __init__(self, model_name: str, params: dict, mode="online"):
        """
        Initialize the Visualizer3D class.

        Parameters:
        - model_name (str): the name of the trained models.
        - params (dict): the training's parameters.
        - entity (str): the WandB entity to use.
        - project (str): the WandB project to use.
        """
        # Mother Class
        super(Visualizer3D, self).__init__(model_name, params, mode=mode)

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
        preds = (preds[0, 1] >= 0.50).float()

        images = {
            f"input_{step}": inputs[0, 0, inputs.shape[2] // 2],
            f"target_{step}": targets[0][0][0],
            f"prediction_{step}": preds[0][0][0],
        }

        for title, image in images.items():
            images[title] = [wandb.Image(image.data)]

        wandb.log(images)
