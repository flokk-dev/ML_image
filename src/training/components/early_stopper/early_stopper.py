"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import typing

# IMPORT: deep learning
import torch


class EarlyStopper:
    def __init__(
            self,
            params: typing.Dict[str, typing.Any]
    ):
        # Attributes
        self._max_duration: int = params["duration"]
        self._current_best: typing.Dict[str, typing.Any] = {
            "loss_value": float("inf"),
            "epoch": None,
            "duration": 0,
            "weights": None,
        }

    def check_epoch(
            self,
            epoch_idx: int,
            loss_value: float,
            model: torch.nn.Module
    ):
        if loss_value < self._current_best["loss_value"]:
            self._current_best["loss_value"] = loss_value
            self._current_best["epoch"] = epoch_idx
            self._current_best["duration"] = 0
            self._current_best["weights"] = model.module.state_dict()
        else:
            self._current_best["duration"] += 1

        if self._current_best["duration"] >= self._max_duration:
            return False
        return True

    def __getitem__(self, key):
        return self._current_best[key]
