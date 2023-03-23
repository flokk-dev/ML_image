"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *


class EarlyStopper:
    """
    Represents an early stopper.

    Attributes
    ----------
        _max_duration : int
            maximum number of epochs without improvement
        _current_best : Dict[str, Any]
            information about the best epoch

    Methods
    ----------
        check_epoch : List[str]
            Checks if the current epoch is the best one, if so updates the early stopper
    """

    def __init__(self, params: Dict[str, Any]):
        """
        Instantiates an EarlyStopper.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour
        """
        # Attributes
        self._max_duration: int = params["duration"]
        self._current_best: Dict[str, Any] = {
            "loss_value": float("inf"),
            "epoch": None,
            "duration": 0,
            "weights": None,
        }

    def check_epoch(self, epoch_idx: int, loss_value: float, weights: dict):
        """
        Checks if the current epoch is the best one, if so updates the early stopper.

        Parameters
        ----------
            epoch_idx : int
                index of the current epoch
            loss_value : float
                loss value during the current epoch
            weights : dict
                weights of the model during the current epoch

        Returns
        ----------
            bool
                file paths within the dataset
        """
        if loss_value < self._current_best["loss_value"]:
            self._current_best["loss_value"] = loss_value
            self._current_best["epoch"] = epoch_idx
            self._current_best["duration"] = 0
            self._current_best["weights"] = weights
        else:
            self._current_best["duration"] += 1

        if self._current_best["duration"] >= self._max_duration:
            return False
        return True

    def __getitem__(self, key) -> Any:
        """
        Parameters
        ----------
            key : str
                key within the early stopper's information dictionary

        Returns
        ----------
            Any
                value associated to the key
        """
        return self._current_best[key]
