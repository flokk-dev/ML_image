"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: deep learning
import torch

# IMPORT: project
from .model import ModelManager
from .loss import LossManager
from .metric import MetricManager

from .early_stopper import EarlyStopper
from .dashboard import Dashboard2D, Dashboard25D, Dashboard3D


class TrainingComponents:
    _DASHBOARDS = {2: Dashboard2D, 2.5: Dashboard25D, 3: Dashboard3D}

    def __init__(self, params, data_info, weights_path):
        # Attributes
        self._params = params

        self._model = self._init_model(data_info, weights_path)
        self._optimizer, self._scheduler = self._init_optimizer_scheduler()

        self._loss = self._init_loss()
        self._metrics = self._init_metrics()

        # Early stopper
        self._early_stopper = EarlyStopper(self._params)

        # Dashboard
        self._dashboard = self._DASHBOARDS[self._params["output_dim"]](
            self._params, train_id="train"
        )

    @property
    def model(self):
        return self._model

    def _init_model(self, data_info, weights_path):
        model = ModelManager()(self._params["model"], data_info, weights_path)
        return torch.nn.DataParallel(model)

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def scheduler(self):
        return self._scheduler

    def _init_optimizer_scheduler(self):
        optimizer = torch.optim.Adam(params=self._model.parameters(), lr=self._params["lr"])
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            optimizer, self._params["lr_multiplier"]
        )

        return optimizer, scheduler

    @property
    def loss(self):
        return self._loss

    def _init_loss(self):
        loss_manager = LossManager()
        return loss_manager(self._params["training_purpose"], self._params["loss"])

    @property
    def metrics(self):
        return self._metrics

    def _init_metrics(self):
        metric_manager = MetricManager()
        return {
            metric_name: metric_manager(self._params["training_purpose"], metric_name)
            for metric_name in self._params["metrics"]
        }

    @property
    def early_stopper(self):
        return EarlyStopper(self._params)

    @property
    def dashboard(self):
        return self._dashboard
