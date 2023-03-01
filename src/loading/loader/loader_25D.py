"""
Creator: HOCQUET Florian, Landza HOUDI
Date: 30/09/2022
Version: 1.0

Purpose: Loads a 2.5D dataset.
"""

from src.loading.loader import Loader
from src.datasets import DataSet25D


class Loader25D(Loader):
    """
    The class representing a 2.5D loader.

    Methods:
    - _gen_loader: Generates a torch.io loader from data.
    - _my_collate: Collects data and yields batch from them.
    """

    def __init__(self, params: dict, path: str):
        """
        Initialize the Loader25D class.

        Parameters:
        - path: the path to the dataset.
        - _params: the loading's parameters.
        - _purpose: the purpose of the loader.
        """
        super(Loader25D, self).__init__(params, path)
        self._dataset = DataSet25D
