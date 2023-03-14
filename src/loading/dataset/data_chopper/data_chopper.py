"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""


class DataChopper:
    def __init__(self):
        pass

    def _chop(self, input_tensor, target_tensor=None):
        raise NotImplementedError()

    def __call__(self, input_tensor, target_tensor=None):
        return self._chop(input_tensor, target_tensor)
