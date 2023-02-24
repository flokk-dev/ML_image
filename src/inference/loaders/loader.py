"""
Creator: HOCQUET Florian, Landza HOUDI
Date: 30/09/2022
Version: 1.0

Purpose: Loads the dataset using its information, allowing to use conditions.
"""

# IMPORT: utils
import os
import glob

from tqdm import tqdm

# IMPORT: data process
import pandas as pd

# IMPORT: deep learning
from torch.utils.data import DataLoader

# IMPORT: project
import paths


class Loader:
    """
    The class representing a loader.

    Attributes:
    - path: the path to the dataset.
    - _params: the loading parameters.
    - _purpose: the purpose of the loader.
    - _load_info: the dataset information.

    Methods:
    - _parse_csv: Parse the csv file as a dictionary.
    - get_data_loader: Returns the data loaders depending on the loader's purpose.
    - _generate_training_paths: Generates the training and validation datasets' paths.
    - _generate_inference_paths: Generates the inference dataset's paths.
    - _gen_loader: Generates a torch data loader from a list of paths.
    - _my_collate: Collects data and formats them.
    """

    def __init__(self, params: dict, path: str):
        """
        Initialize the Loader class.

        Parameters:
        - path: the path to the dataset.
        - _params: the loading's parameters.
        - _purpose: the purpose of the loader.
        """
        self._params = params
        self.path = path

        self._dataset = None
        self._load_info = self._parse_csv(os.path.join(self.path, paths.INFERENCE_INFO_PATH))

    @staticmethod
    def _parse_csv(path: str) -> dict:
        """
        Parse the csv file as a dictionary.

        Parameters:
        - path (str): the csv file's path.

        Returns:
        - (dict): the dict containing the csv file's information.
        """
        df = pd.read_csv(path)

        info = df.set_index("patient_id")[["shape", "PatientSize", "PatientWeight"]].T.to_dict()
        for key, value in info.items():
            value["shape"] = list(map(int, value["shape"][1:-1].split(",")))
            info[key]["IMC"] = value["PatientWeight"] / value["PatientSize"]**2

        return info

    def get_data_loader(self):
        """
        Returns the data loaders depending on the loader's purpose.

        Returns:
            - ((dict, dict)): the training and validation data loaders.
        """
        print("\nRépartition des chemins de fichiers.")
        inference_paths = self._generate_paths()

        return self._gen_loader(inference_paths)

    def _generate_paths(self) -> dict:
        """
        Generates the inference dataset's paths.

        Returns:
        - (dict): the inference dataset's paths.
        """
        # Split the paths
        inference_info = {"inputs": list(), "targets": list(), "IMC": list()}

        cpt = 0
        for patient_id, patient_info in tqdm(self._load_info.items()):
            input_path, target_path = sorted(
                os.listdir(os.path.join(self.path, str(patient_id)))
            )[-2:]

            input_path = os.path.join(self.path, str(patient_id), input_path, "TEP.npy")
            inference_info["inputs"].append(input_path)

            target_path = os.path.join(self.path, str(patient_id), target_path, "TARGET.npy")
            inference_info["targets"].append(target_path)

            inference_info["IMC"].append(patient_info["IMC"])

            cpt += 1
            if cpt > self._params["nb_data"]:
                break

        return inference_info

    def _gen_loader(self, info) -> DataLoader:
        """
        Generates a torch data loader from a list of paths.

        Parameters:
        - data (list): the data to generate loader from.
        """
        dataset = self._dataset(self._params, self._load_info, info)
        return DataLoader(dataset, batch_size=1, collate_fn=self._my_collate)

    def _my_collate(self, data: list) -> tuple:
        """
        Collects data and formats them.

        Parameters:
        - data (list): the data to generate batch from.

        Returns:
        - (torch.Tensor): the output tensor.
        """
        return data[0][0], data[0][1], data[0][2]
