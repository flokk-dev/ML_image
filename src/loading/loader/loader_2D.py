"""
Creator: HOCQUET Florian, Landza HOUDI
Date: 30/09/2022
Version: 1.0

Purpose: Loads a 2D dataset.
"""

from src.loading.loader import Loader
from src.datasets import DataSet2D


class Loader2D(Loader):
    """
    The class representing a 2D loader.

    Methods:
    - _gen_loader: Generates a torch.io loader from data.
    - _my_collate: Collects data and yields batch from them.
    """

    def __init__(self, params: dict, path: str):
        """
        Initialize the Loader2D class.

        Parameters:
        - path: the path to the dataset.
        - _params: the loading's parameters.
        - _purpose: the purpose of the loader.
        """
        super(Loader2D, self).__init__(params, path)
        self._dataset = DataSet2D
