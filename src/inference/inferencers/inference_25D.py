"""
Creator: HOCQUET Florian, Landza HOUDI
Date: 30/09/2022
Version: 1.0

Purpose: Computes the inference of a given 2.5D model.
"""

# IMPORT: deep learning
import torch
from torch import nn

# IMPORT: projet
import paths

from .inferencer import Inference

from src.inference.loaders import Loader25D
from src.model import UNetR25D, AttentionUnet25D, SwinUNETR25D


class Inference25D(Inference):
    """
    The class computing the inference for a 2.5D model.

    Attributes:
    - _model: the model on which to compute the inference.
    - _loader: the inference dataset.
    """
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, params: dict, data_path: str, model_path: str):
        """
        Initialize the Inference25D class.

        Parameters:
        - data_path: the path to the dataset.
        - model_path: the path to the model's weights dict.
        - params: the inference's parameters.
        """
        super(Inference25D, self).__init__(params, data_path, model_path)

        # Loading trained model for inference
        self._model = SwinUNETR25D(weights_path=model_path, in_channels=self._params["patch_height"])
        self._model.eval()
        self._model = nn.DataParallel(self._model).to(torch.device(self._DEVICE))

        # Loader for inference's datas
        self._loader = Loader25D(params, data_path)

    def _predict(self, input_tensor):
        """
        Computes the prediction using a model.

        Parameters:
        - input (torch.Tensor): the tensor to use for the prediction.

        Returns:
        - (torch.Tensor): the predicted tensor.
        """
        # Clear GPU cache
        torch.cuda.empty_cache()

        # PREDICTION
        prediction = torch.Tensor()

        batch_size = self._params["batch_size"]
        for i in range(0, input_tensor.shape[0], batch_size):
            batch = input_tensor[i: i + batch_size].type(torch.float32).to(torch.device(self._DEVICE))

            logits = self._model(batch).detach()
            prediction = torch.cat((prediction, logits.cpu()), dim=0)

        shape = list(prediction.shape)
        shape[0] = self._params["patch_height"] // 2

        return torch.cat((torch.zeros(shape), prediction, torch.zeros(shape)), dim=0)
