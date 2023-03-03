"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import os


class LoadingManager:
    def __init__(self, params: dict, path: str):
        # Attributes
        self._params = params
        self._path = path
