"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import time
from tqdm import tqdm

# IMPORT: deep learning
import torch

# IMPORT: project
from .early_stopper import EarlyStopper
from .dashboard import Dashboard2D, Dashboard25D, Dashboard3D


class Trainer:
    _DASHBOARDS = {2: Dashboard2D, 2.5: Dashboard25D, 3: Dashboard3D}

    def __init__(self, params):
        # Attributes
        self._params = params
        self._name = None

        # Components
        self._data_loaders = None
        self._learner = None

        self._early_stopper = EarlyStopper(self._params)
        self._dashboard = self._DASHBOARDS[self._params["input_dim"]](
            self._params, self._name
        )

    def _launch(self):
        print("\nLancement de l'entrainement.")

        time.sleep(1)
        for epoch in tqdm(self._params["nb_epochs"]):
            # Clear cache
            torch.cuda.empty_cache()

            # Learn
            self._run_epoch(step="train")
            self._run_epoch(step="valid")

            # Update the epoch
            self._dashboard.upload_values(self._learner.scheduler.get_last_lr()[0])
            self._learner.scheduler.step()

        # End the training
        time.sleep(30)
        self._dashboard.shutdown()

    def _run_epoch(self, step):
        raise NotImplementedError()

    def __call__(self):
        # Clean cache
        torch.cuda.empty_cache()

        self._launch()
