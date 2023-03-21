"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: project
from .loss import Loss

from .classification import \
    CELoss, BCELoss, DiceLoss, \
    DiceCELoss, DiceFocalLoss, FocalLoss, FocalTverskyLoss

from .regression import MAELoss, MSELoss, RMSELoss, HuberLoss


class LossManager(dict):
    def __init__(self):
        super(LossManager, self).__init__({
            "classification": {
                "cross entropy": CELoss,
                "binary cross entropy": BCELoss,
                "dice": DiceLoss,
                "dice cross entropy": DiceCELoss,
                "dice focal": DiceFocalLoss,
                "focal": FocalLoss,
                "focal tversky": FocalTverskyLoss,
            },
            "regression": {
                "mean absolute error": MAELoss,
                "mean square error": MSELoss,
                "root mean square error": RMSELoss,
                "huber": HuberLoss
            }
        })

    def __call__(
            self,
            training_purpose: str,
            loss_id: str
    ) -> Loss:
        try:
            return self[training_purpose][loss_id]()
        except KeyError:
            raise KeyError(f"The {loss_id} isn't handled by the metric manager.")
