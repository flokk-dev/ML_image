"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""


class DataChopper:
    def __init__(self):
        # Attributes
        self._patch_height = 5

    def _chopping(self, input_tensor, target_tensor=None):
        raise NotImplementedError()

    def __call__(self, input_tensor, target_tensor=None):
        return self._chopping(input_tensor, target_tensor)
