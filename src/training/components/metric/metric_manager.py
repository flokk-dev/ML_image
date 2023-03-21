"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: project
from .metric import Metric

from .classification import Accuracy, JaccardIndex, Precision, Recall, F1Score
from .regression import MAE, MSE, RMSE, PSNR, SSIM


class MetricManager(dict):
    def __init__(self):
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

    def __call__(
            self,
            training_purpose: str,
            metric_id: str
    ) -> Metric:
        try:
            return self[training_purpose][metric_id]()
        except KeyError:
            raise KeyError(f"The {metric_id} isn't handled by the metric manager.")
