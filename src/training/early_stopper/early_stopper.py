"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""


class EarlyStopper:
    def __init__(self, params):
        # Attributes
        self._max_duration = params["duration"]
        self._current_best = {
            "loss_value": float("inf"),
            "epoch": 0,
            "duration": 0,
            "weights": dict(),
        }

    def check_epoch(self, epoch_idx, loss_value, model):
        if loss_value < self._current_best["loss_value"]:
            self._current_best["loss"] = loss_value
            self._current_best["epoch"] = epoch_idx
            self._current_best["duration"] = 0
            self._current_best["weights"] = model.module.state_dict()
        else:
            self._current_best["duration"] += 1

        if self._current_best["duration"] >= self._max_duration:
            return False
        return True