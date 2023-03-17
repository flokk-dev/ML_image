"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import os

# IMPORT: deep learning
import torch

# IMPORT: visualization
import wandb


class Visualizer:
    def __init__(self, params, model_name, mode="online"):
        # Initialize the wandb entity
        os.environ["WANDB_SILENT"] = "true"
        wandb.init(entity="machine_learning_lib", project="test", mode=mode)

        wandb.run.name = model_name
        wandb.config = params

        # Attributes
        self._loss = {"train": list(), "valid": list()}
        self._metrics = {m_name: {"train": list(), "valid": list()} for m_name in []}

    @staticmethod
    def collect_info(model):
        wandb.watch(model)

    @staticmethod
    def plot_res(loss: dict, metrics: dict, epoch: int, lr: list):
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
        raise NotImplementedError()

    @staticmethod
    def finish_run():
        wandb.finish()
