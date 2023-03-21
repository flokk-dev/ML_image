"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import os

# IMPORT: visualization
import wandb


class Dashboard:
    def __init__(self, params, train_id, mode="online"):
        # Initialize the wandb entity
        os.environ["WANDB_SILENT"] = "true"
        wandb.init(entity="machine_learning_lib", project="test", mode=mode)

        wandb.run.name = train_id
        wandb.config = params

        # Attributes
        self._loss = {"train": list(), "valid": list()}
        self._metrics = {
            metric_name: {"train": list(), "valid": list()}
            for metric_name in params["metrics"]
        }

    def update_loss_metrics(self, loss, metrics, step):
        # Update the loss
        current_loss = sum(loss) / len(loss)
        self._loss[step].append(current_loss)

        # Update the metrics
        for metric_name, metric_values in metrics.items():
            self._metrics[metric_name][step].append(
                sum(metrics[metric_name]) / len(metrics[metric_name])
            )

    @staticmethod
    def collect_info(model):
        wandb.watch(model)

    def upload_values(self, lr):
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
    def upload_images(inputs, targets, predictions, step):
        raise NotImplementedError()

    @staticmethod
    def shutdown():
        wandb.finish()
