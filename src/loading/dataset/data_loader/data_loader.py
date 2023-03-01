"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""


class DataLoader:
    def __init__(self):
        pass

    def _load(self, path):
        raise NotImplementedError()

    def __call__(self, path):
        self._load(path)
