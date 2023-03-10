"""
Creator: HOCQUET Florian, LANDZA Houdi
Date: 30/09/2022
Version: 1.0

Purpose: Trains a given 3D model.
"""

# IMPORT: data processing

# IMPORT: deep learning
from torch import optim

# IMPORT: project
import utils

from .trainer import Trainer

from src.train.loaders import Loader3D
from src.model import AttentionUnet3D, SwinUNETR3D
from src.visualizers import Visualizer3D


class Trainer3D(Trainer):
    """
    The class training a given 3D model.

    Attributes:
    - _model (Model): the model to train.
    - _name (str): the training's name.
    - _loader: the loading manager.
    - _loss: the loss to minimize.
    - _optimizer: the training's optimizer.
    - _scheduler: the sheduler managing the learning rate's evolution.
    - _visualizer (Visualiser): the training's visualizer.

    Methods
    - _train_block: The training's function.
    """

    def __init__(self, params: dict, data_path: str, model_path: str = None):
        """
        Initialize the Trainer3D class.

        Parameters:
        - data_path (Model): the path to the dataset.
        - params (dict): the training's parameters.
        - model_path (bool): the path to the model's weights dict.
        - tuning (bool): if True, tune the model.
        """
        # Mother Class
        super(Trainer3D, self).__init__(params)

        # Model and data
        self._model = AttentionUnet3D(model_path)
        self._loader = Loader3D(params, data_path)
        
        self._name = f"{utils.get_datetime()}_{self._model.name}"

        # Parameters
        self._optimizer = optim.Adam(params=self._model.parameters(), lr=self._params["lr"])

        lr_multiplier = lambda epoch: self._params["lr_multiplier"]
        self._scheduler = optim.lr_scheduler.MultiplicativeLR(self._optimizer, lr_multiplier)

        # Launch Visualisation
        self._visualizer = Visualizer3D(self._name, params=self._params, mode="online")
        self._visualizer.collect_info(self._model)
