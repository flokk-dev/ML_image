"""
Creator: HOCQUET Florian, Landza HOUDI
Date: 30/09/2022
Version: 2.0

Purpose: Computes the inference of a given model.
"""

# IMPORT: utils
import os
import csv

from tqdm import tqdm

# IMPORT: data process
import numpy as np

# IMPORT: deep learning
import torch
from torch.nn import L1Loss, MSELoss

# IMPORT: projet
import paths
import utils

from src.metrics import RMSE, SAE


class Inference:
    """
    The class computing the inference.

    Attributes:
    - _params: the inference's parameters.
    - _model: the model on which to compute the inference.
    - _loader: the inference dataset.
    - _metrics: the inference's metrics.
    - _results: the inference's results.

    Methods:
    - launch: Computes the different inference's metrics.
    - _predict: Computes the prediction using a model.
    - _save: Saves the results in a csv file.
    """
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, params: dict, data_path: str, model_path: str):
        """
        Initialize the Inference class.

        Parameters:
        - params: the inference's parameters.
        """
        # Parameters
        self._params = params
        self._data_path = data_path
        self._model_path = model_path

        # Inference
        self._model = None

        self._loader = None
        self._data_loader = None

        # Metrics
        self._metrics = {
            "SAE": SAE().to(torch.device(self._DEVICE)),
            "L1Loss": L1Loss().to(torch.device(self._DEVICE)),
            "MSELoss": MSELoss().to(torch.device(self._DEVICE)),
            "RMSE": RMSE().to(torch.device(self._DEVICE))
        }

        # Results
        self._results = {m: list() for m in self._metrics}

    def launch(self):
        """
        Launch the inference.
        """
        print()
        self._data_loader = self._loader.get_data_loader()

        print("\nCalcul de l'inférence:")
        self._compute_inference()

        for metric, results in self._results.items():
            m_t_median = round(np.quantile(self._results[metric], q=0.5), 5)
            m_t_mean = round(np.mean(self._results[metric]), 5)
            m_t_std = round(np.std(self._results[metric]), 5)

            print(f"{metric}: median={m_t_median}, mean={m_t_mean}, std={m_t_std}")
            self._results[metric] = str(m_t_mean).replace(".", ","), str(m_t_std).replace(".", ",")

        self._save()

    def _compute_inference(self):
        """
        Computes metrics over all the test dataset.
        """
        batch_idx = 0
        for batch in tqdm(self._data_loader):
            # Clear GPU cache
            torch.cuda.empty_cache()

            prediction = self._predict(batch[0])
            target = batch[1]

            # LOSS AND METRIC
            for metric, results in self._results.items():
                self._results[metric].append(
                    self._metrics[metric](
                        prediction.to(torch.device(self._DEVICE)),
                        target.to(torch.device(self._DEVICE))
                    ).item()
                )

            batch_idx += 1

    def _predict(self, input_tensor):
        """
        Computes the prediction using a model.

        Parameters:
        - input_tensor (torch.Tensor): the tensor to use for the prediction.

        Returns:
        - (torch.Tensor): the predicted tensor.
        """
        raise NotImplementedError()

    def _save(self):
        """
        Saves the results.
        """
        self._save_results()

    def _save_results(self):
        """
        Saves the results in a csv file.
        """
        info = dict()

        info["model_path"] = self._model.module.weights_path
        info["data_path"] = self._loader.path
        info["number_of_params"] = self.count_parameters()

        for metric, results in self._results.items():
            info[f"{metric}"] = results

        if not os.path.exists(paths.INFERENCE_RES_PATH):
            with open(paths.INFERENCE_RES_PATH, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(list(info.keys()))

        with open(paths.INFERENCE_RES_PATH, "a", newline="", encoding="utf-8") as f:
            dictwriter_object = csv.DictWriter(f, fieldnames=list(info.keys()))
            dictwriter_object.writerow(info)
            f.close()

    def count_parameters(self):
        return sum(p.numel() for p in self._model.parameters() if p.requires_grad)
