"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import typing
import os


class Loader:
    def __init__(
            self, 
            params: typing.Dict[str, typing.Any]
    ):
        """
        pass.
        """
        # Attributes
        self._params: typing.Dict[str, typing.Any] = params
        self._input_paths: typing.Dict[str, typing.List[str]] = {"train": list(), "valid": list()}

    def _extract_paths(
            self,
            dataset_path: str
    ) -> typing.List[str]:
        """
        pass.
        """
        file_paths: typing.List[str] = list()
        for root, dirs, files in os.walk(dataset_path, topdown=False):
            for file_path in map(lambda e: os.path.join(root, e), files):
                if self._file_depth(file_path, self._params["dataset_name"]) == self._params["file_depth"]:
                    file_paths.append(file_path)

        return list(sorted(file_paths))

    @staticmethod
    def _file_depth(
            path: str,
            dataset_name: str
    ) -> int:
        """
        pass.
        """
        # try to find "dataset_name" in path
        try:
            idx: int = path.index(f"{os.sep}{dataset_name}{os.sep}")
            return len(path[idx + 1:].split(os.sep)) - 2
        except ValueError:
            # try to find if dataset_name begin the path
            try:
                idx: int = path.index(f"{dataset_name}{os.sep}")
                return len(path[idx + 1:].split(os.sep)) - 2
            except ValueError:
                raise ValueError(f"\"{dataset_name}\" n'apparait pas dans le chemin spécifié.")

    def _generate_data_loaders(self) -> typing.Any:
        """
        pass.
        """
        raise NotImplementedError()

    def __call__(
            self,
            dataset_path
    ) -> typing.Any:
        """
        pass.
        """
        raise NotImplementedError()
