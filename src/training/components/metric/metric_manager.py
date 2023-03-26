"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *

# IMPORT: project
from .metric import Metric

from .classification import Accuracy, JaccardIndex, Precision, Recall, F1Score
from .regression import MAE, MSE, RMSE, PSNR, SSIM


class MetricManager(dict):
    """
    Represents a metric manager.

    Attributes
    ----------
        _params : Dict[str, int]
            parameters needed to adjust the metrics behaviour
    """

    def __init__(self, params: Dict[str, int]):
        """
        Instantiates a MetricManager.

        Parameters
        ----------
            params : Dict[str, int]
                parameters needed to adjust the metrics behaviour
        """
        # Mother class
        super(MetricManager, self).__init__({
            "classification": {
                "accuracy": Accuracy,
                "jaccard index": JaccardIndex,
                "precision": Precision,
                "recall": Recall,
                "f1score": F1Score
            },
            "regression": {
                "mean absolute error": MAE,
                "mean square error": MSE,
                "root mean square error": RMSE,
                "peak signal noise ratio": PSNR,
                # "structural similarity index measure": SSIM
            }
        })

        # Attributes
        self._params: Dict[str, int] = params

    def __call__(self, training_purpose: str, metric_id: str) -> Metric:
        """
        Parameters
        ----------
            training_purpose : str
                purpose of the training
            metric_id : str
                id of the loss

        Returns
        ----------
            Loss
                loss function associated with the loss id

        Raises
        ----------
            KeyError
                loss id isn't handled by the loss manager
        """
        try:
            return self[training_purpose][metric_id](self._params)
        except KeyError:
            raise KeyError(f"The {metric_id} isn't handled by the metric manager.")
