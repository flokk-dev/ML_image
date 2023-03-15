"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: deep learning
import torch
from torch import optim

# IMPORT: project
from .early_stopper import EarlyStopper
from . import model, loss, metric


class TrainingComponents:
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, params):
        # Attributes
        self._params = params

        # Model
        self._model = self._init_model()

        # Early stopping and checkpoints
        self._early_stopper = EarlyStopper()

        # Optimizer
        self._optimizer = optim.Adam(params=self._model.parameters(), lr=self._params["lr"])
        self._scheduler = optim.lr_scheduler.MultiplicativeLR(
            self._optimizer, self._params["lr_multiplier"]
        )

        # Loss and metrics
        self._loss = self._init_loss()
        self._metrics = self._init_metrics()

        # Visualizer
        self._visualizer = None

    def _init_model(self):
        possible_models = {
            "unet": {2: model.UNet2D, 2.5: model.UNet25D, 3: model.UNet3D},
            "attention_unet": {2: model.AttentionUNet2D, 2.5: model.AttentionUNet25D, 3: model.AttentionUNet3D},
            "unet_r": {2: model.UNetR2D, 2.5: model.UNetR25D, 3: model.UNetR3D},
            "swin_unet_r": {2: model.SwinUNetR2D, 2.5: model.SwinUNetR2D, 3: model.SwinUNetR3D},
        }

        return possible_models[self._params["model"]][self._params["output_dim"]]()

    def _init_loss(self):
        possible_losses = {
            "cross_entropy": loss.CELoss,
            "binary_cross_entropy": loss.BCELoss,
            "dice": loss.DiceLoss,
            "dice_cross_entropy": loss.DiceCELoss,
            "dice_focal": loss.DiceFocalLoss,
            "focal_loss": loss.FocalLoss,
            "focal_tversky": loss.FocalTverskyLoss,
            "unified_focal": loss.UnifiedFocalLoss,
            "mean_absolute_error": loss.MAELoss,
            "mean_square_error": loss.MSELoss,
            "root_mean_square_error": loss.RMSELoss,
            "huber": loss.HuberLoss
        }

        return possible_losses[self._params["loss"]]()

    def _init_metrics(self):
        possible_metrics = {
            "accuracy": metric.Accuracy,
            "jaccard_index": metric.JaccardIndex,
            "f1_score": metric.JaccardIndex,
            "precision": metric.Precision,
            "recall": metric.Recall,
            "mean_absolute_error": metric.MAE,
            "mean_square_error": metric.MSE,
            "root_mean_square_error": metric.RMSE,
            "peak_signal_noise_ratio": metric.PSNR,
            "structural_similarity_index_measure": metric.SSIM
        }

        metrics = dict()
        for metric_name in self._params["metrics"]:
            metrics[metric_name] = possible_metrics[metric_name]

        return metrics

    @property
    def early_stopper(self):
        return self._early_stopper

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def scheduler(self):
        return self._scheduler

    @property
    def loss(self):
        return self._loss

    @property
    def metrics(self):
        return self._metrics

    @property
    def visualizer(self):
        return self._visualizer
