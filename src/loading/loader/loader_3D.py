"""
Creator: HOCQUET Florian, Landza HOUDI
Date: 30/09/2022
Version: 1.0

Purpose: Loads a 3D dataset.
"""

from src.loading.loader import Loader
from src.datasets import DataSet3D


class Loader3D(Loader):
    """
    The class representing a 3D loader.

    Methods:
    - _gen_loader: Generates a torch.io loader from data.
    - _my_collate: Collects data and yields batch from them.
    """

    def __init__(self, params: dict, path: str):
        """
        Initialize the Loader3D class.

        Parameters:
        - path: the path to the dataset.
        - _params: the loading's parameters.
        - _purpose: the purpose of the loader.
        """
        super(Loader3D, self).__init__(params, path)
        self._dataset = DataSet3D
