"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: dataset visualization
import wandb

# IMPORT: project
from .dashboard import Dashboard


class Dashboard2D(Dashboard):
    def __init__(self, params, train_id, mode="online"):
        # Mother Class
        super(Dashboard2D, self).__init__(params, train_id, mode=mode)

    @staticmethod
    def upload_images(inputs, targets, predictions, step):
        images = {
            f"input_{step}": inputs[0],
            f"target_{step}": targets[0],
            f"prediction_{step}": predictions[0],
        }

        for title, image in images.items():
            images[title] = [wandb.Image(image.data)]

        wandb.log(images)


class Dashboard25D(Dashboard):
    def __init__(self, params, train_id, mode="online"):
        # Mother Class
        super(Dashboard25D, self).__init__(params, train_id, mode=mode)

    @staticmethod
    def upload_images(inputs, targets, predictions, step):
        images = {
            f"input_{step}": inputs[0, inputs.shape[1] // 2],
            f"target_{step}": targets[0][0],
            f"prediction_{step}": predictions[0][0],
        }

        for title, image in images.items():
            images[title] = [wandb.Image(image.data)]

        wandb.log(images)


class Dashboard3D(Dashboard):
    def __init__(self, params, train_id, mode="online"):
        # Mother Class
        super(Dashboard3D, self).__init__(params, train_id, mode=mode)

    @staticmethod
    def upload_images(inputs, targets, predictions, step):
        predictions = (predictions[0, 1] >= 0.50).float()

        images = {
            f"input_{step}": inputs[0, 0, inputs.shape[2] // 2],
            f"target_{step}": targets[0][0][0],
            f"prediction_{step}": predictions[0][0][0],
        }

        for title, image in images.items():
            images[title] = [wandb.Image(image.data)]

        wandb.log(images)
