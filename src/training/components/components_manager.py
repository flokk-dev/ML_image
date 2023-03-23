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
import utils

from .model import ModelManager

from .loss import Loss, LossManager
from .metric import Metric, MetricManager

from .early_stopper import EarlyStopper
from .dashboard import Dashboard, Dashboard2D, Dashboard25D, Dashboard3D


class TrainingComponentsManager:
    """
    Represents a general components manager, that will be derived depending on the use case.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour
        _model : torch.nn.Module
            training's model
        _optimizer : torch.optim.Optimizer
            model's optimizer
        _scheduler : torch.nn.Module
            optimizer's scheduler
        _loss : Loss
            training's loss function
        _metrics : Dict[str, Metric]
            training's metrics
        _early_stopper : EarlyStopper
            early stopping method allowing to end the training when required
        _dashboard : Dashboard
            dashboard allowing to visualize the training's progression
    """

    _DASHBOARDS = {2: Dashboard2D, 2.5: Dashboard25D, 3: Dashboard3D}

    def __init__(self, params: Dict[str, Any], data_info: Dict[str, int], weights_path: str):
        """
        Instantiates a TrainingComponentsManager.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour
            data_info : Dict[str, int]
                information about the data within the dataset
            weights_path : str
                path to the model's weights
        """
        # Attributes
        self._params: Dict[str, Any] = params

        # Model
        self._model: torch.nn.Module = self._init_model(data_info, weights_path)

        # Optimizer and scheduler
        self._optimizer: torch.optim.Optimizer = None
        self._scheduler: torch.nn.Module = None

        self._optimizer, self._scheduler = self._init_optimizer_scheduler()

        # Loss and metrics
        self._loss: Loss = self._init_loss()
        self._metrics: Dict[str, Metric] = self._init_metrics()

        # Early stopper
        self._early_stopper: EarlyStopper = EarlyStopper(self._params, self._loss.behaviour)

        # Dashboard
        self._dashboard: Dashboard = self._DASHBOARDS[self._params["output_dim"]](
            self._params, train_id=f"{utils.get_datetime()}_{self._model}"
        )

    @property
    def model(self) -> torch.nn.Module:
        """
        Returns the training's model.

        Returns
        ----------
            torch.nn.Module
                training's model
        """
        return self._model

    def _init_model(self, data_info, weights_path) -> torch.nn.Module:
        """
        Initializes the training's model.

        Returns
        ----------
            torch.nn.Module
                training's model
        """
        model = ModelManager()(self._params["model"], data_info, weights_path)
        return torch.nn.DataParallel(model)

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        """
        Returns the model's optimizer.

        Returns
        ----------
            torch.nn.Module
                model's optimizer
        """
        return self._optimizer

    @property
    def scheduler(self) -> torch.nn.Module:
        """
        Returns the optimizer's scheduler.

        Returns
        ----------
            torch.nn.Module
                optimizer's scheduler
        """
        return self._scheduler

    def _init_optimizer_scheduler(self) -> Tuple[torch.optim.Optimizer, torch.nn.Module]:
        """
        Initializes the model's optimizer and optimizer's scheduler.

        Returns
        ----------
            Tuple[torch.optim.Optimizer, torch.nn.Module]
                model's optimizer and optimizer's scheduler
        """
        optimizer: torch.optim.Optimizer = torch.optim.Adam(
            params=self._model.parameters(), lr=self._params["lr"]
        )
        scheduler: torch.nn.Module = torch.optim.lr_scheduler.MultiplicativeLR(
            optimizer, self._params["lr_multiplier"]
        )

        return optimizer, scheduler

    @property
    def loss(self) -> Loss:
        """
        Returns the training's loss function.

        Returns
        ----------
            torch.nn.Module
                training's loss function
        """
        return self._loss

    def _init_loss(self) -> Loss:
        """
        Initializes the training's loss function.

        Returns
        ----------
            torch.nn.Module
                training's loss function
        """
        loss_manager: LossManager = LossManager()
        return loss_manager(self._params["training_purpose"], self._params["loss"])

    @property
    def metrics(self) -> Dict[str, Metric]:
        """
        Returns the training's metrics.

        Returns
        ----------
            Dict[str, Metric]
                training's metrics
        """
        return self._metrics

    def _init_metrics(self) -> Dict[str, Metric]:
        """
        Initializes the training's metrics.

        Returns
        ----------
            torch.nn.Module
                training's metrics
        """
        metric_manager: MetricManager = MetricManager()
        return {
            metric_name: metric_manager(self._params["training_purpose"], metric_name)
            for metric_name in self._params["metrics"]
        }

    @property
    def early_stopper(self) -> EarlyStopper:
        """
        Returns the early stopping method allowing to end the training when required.

        Returns
        ----------
            EarlyStopper
                early stopping method allowing to end the training when required
        """
        return EarlyStopper(self._params)

    @property
    def dashboard(self) -> Dashboard:
        """
        Returns the dashboard allowing to visualize the training's progression.

        Returns
        ----------
            Dashboard
                dashboard allowing to visualize the training's progression
        """
        return self._dashboard
