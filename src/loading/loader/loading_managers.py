"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: project
from .loading_manager import Loading


class LoadingUnsupervised(Loading):
    def __init__(self, params):
        # Mother Class
        super(LoadingUnsupervised, self).__init__(params)

    def _order_paths(self, file_paths):
        for idx in range(len(file_paths)):
            self._input_paths.append(file_paths[idx])

    def _generate_dataset(self):
        return self._datasets[self._params["input_dim"]](
            self._params, self._input_paths
        )

    def __call__(self, path):
        super().__call__(path)

        self._extract_path(path)
        return self._generate_data_loader()


class LoadingSupervised(Loading):
    def __init__(self, params):
        # Mother Class
        super(LoadingSupervised, self).__init__(params)

        # Attributes
        self._target_paths = list()

    def _order_paths(self, file_paths):
        for idx in range(0, len(file_paths), 2):
            self._input_paths.append(file_paths[idx])
            self._target_paths.append(file_paths[idx+1])

    def _generate_dataset(self):
        return self._datasets[self._params["input_dim"]](
            self._params, self._input_paths, self._target_paths
        )

    def __call__(self, path):
        super().__call__(path)
        self._target_paths = list()

        self._extract_path(path)
        return self._generate_data_loader()
