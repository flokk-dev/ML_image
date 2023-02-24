"""
Creator: HOCQUET Florian, Landza HOUDI
Date: 30/09/2022
Version: 1.0

Purpose: Computes the inference of a given 3D model.
"""

# IMPORT: deep learning
import torch
from torch import nn

import torchio
from torchio.data import GridAggregator

# IMPORT: projet
import paths
import utils

from .inferencer import Inference

from src.inference.loaders import Loader3D
from src.model import SwinUNETR25D, SwinUNETR3D


class Inference3D(Inference):
    """
    The class computing the inference for a 3D model.

    Attributes:
    - _model: the model on which to compute the inference.
    - _loader: the inference dataset.
    """
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, params: dict, data_path: str, model_path: str):
        """
        Initialize the Inference3D class.

        Parameters:
        - data_path: the path to the dataset.
        - model_path: the path to the model's weights dict.
        - params: the inference's parameters.
        """
        super(Inference3D, self).__init__(params, data_path, model_path)

        # Loading trained model for inference
        self._model = SwinUNETR3D(weights_path=model_path)
        self._model.eval()
        self._model = nn.DataParallel(self._model).to(torch.device(self._DEVICE))

        # Loader for inference's datas
        self._loader = Loader3D(params, data_path)

    def _predict(self, input_tensor):
        """
        Computes the prediction using a model.

        Parameters:
        - input_tensor (torch.Tensor): the tensor to use for the prediction.

        Returns:
        - (torch.Tensor): the predicted tensor.
        """
        # Clear GPU cache
        torch.cuda.empty_cache()

        # PREDICTION
        prediction = torch.Tensor()

        batch_size = self._params["batch_size"]
        for i in range(0, input_tensor[0].shape[0], batch_size):
            batch = input_tensor[0][i: i + batch_size].type(torch.float32).to(torch.device(self._DEVICE))

            logits = self._model(batch)
            prediction = torch.cat((prediction, logits.cpu()), dim=0)

        shape = input_tensor[2]
        reshaped_prediction = torch.zeros((prediction.shape[1], shape[1], shape[2], shape[3]))

        cpt = 0
        z, x, y = input_tensor[1]
        for i in range(0, input_tensor[2][1], z):
            for j in range(0, input_tensor[2][2], x):
                for k in range(0, input_tensor[2][3], y):
                    reshaped_prediction[:, i:i+z, j:j+x, k:k+y] = prediction[cpt]
                    cpt += 1

        reshaped_prediction = torch.movedim(reshaped_prediction, 1, 0)
        return reshaped_prediction[:input_tensor[3]]
