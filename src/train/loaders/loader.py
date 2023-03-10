"""
Creator: HOCQUET Florian, Landza HOUDI
Date: 30/09/2022
Version: 1.0

Purpose: Loads the dataset using its information, allowing to use conditions.
"""

# IMPORT: utils
import os
import glob
import random

from tqdm import tqdm

# IMPORT: data process
import pandas as pd

# IMPORT: deep learning
import torch
from torch.utils.data import DataLoader, TensorDataset

# IMPORT: project
import utils
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
        self._load_info = self._parse_csv(os.path.join(self.path, paths.TRAIN_INFO_PATH))

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

        info = df.set_index("patient_id")[["shape"]].T.to_dict()
        for key, value in info.items():
            value["shape"] = list(map(int, value["shape"][1:-1].split(",")))

        return {k: info[k] for k in random.sample(list(info.keys()), len(list(info.keys())))}

    def get_data_loader(self):
        """
        Returns the data loaders depending on the loader's purpose.

        Returns:
            - ((dict, dict)): the training and validation data loaders.
        """
        print("\nR??partition des chemins de fichiers.")
        training_paths, validation_paths = self._generate_paths()

        print("\nChargements des donn??es depuis la base.")
        return self._gen_loader(training_paths), self._gen_loader(validation_paths)

    def _generate_paths(self) -> tuple:
        """
        Generates the training and validation datasets' paths.

        Returns:
            - ((dict, dict)): the training and validation datasets' paths.
        """
        # Verify parameters
        if self._params["nb_data"] >= len(self._load_info):
            self._params["nb_data"] = len(self._load_info)

        nb_train = int((1 - self._params["valid_coeff"]) * self._params["nb_data"])

        # Split the paths
        train_paths = {"inputs": list(), "targets": list(), "shapes": list()}
        valid_paths = {"inputs": list(), "targets": list(), "shapes": list()}

        cpt = 0

        for_train = True
        for patient_id, patient_info in tqdm(self._load_info.items()):
            input_path, target_path = sorted(
                os.listdir(os.path.join(self.path, str(patient_id)))
            )[-2:]

            input_path = os.path.join(self.path, str(patient_id), input_path, "TEP.npy")
            target_path = os.path.join(self.path, str(patient_id), target_path, "TARGET.npy")

            if for_train:
                train_paths["inputs"].append(input_path)
                train_paths["targets"].append(target_path)
                train_paths["shapes"].append(patient_info["shape"])
            else:
                valid_paths["inputs"].append(input_path)
                valid_paths["targets"].append(target_path)
                valid_paths["shapes"].append(patient_info["shape"])

            cpt += 1
            if for_train and cpt >= nb_train:
                for_train = False

            if cpt > self._params["nb_data"]:
                break

        return train_paths, valid_paths

    def _gen_loader(self, paths) -> DataLoader:
        """
        Generates a torch data loader from a list of paths.

        Parameters:
        - data (list): the data to generate loader from.
        """
        dataset = self._dataset(self._params, paths)

        if self._params["lazy_loading"]:
            return DataLoader(dataset, batch_size=25, shuffle=True, collate_fn=self._my_collate)
        else:
            tensors = list(zip(*[batch for batch in tqdm(DataLoader(dataset))]))

            inputs = utils.remove_dim(torch.cat(tensors[0], dim=1))
            targets = utils.remove_dim(torch.cat(tensors[1], dim=1))

            return DataLoader(TensorDataset(inputs, targets), num_workers=self._params["workers"],
                              batch_size=self._params["batch_size"], shuffle=True, drop_last=True)

    def _my_collate(self, data: list) -> tuple:
        """
        Collects data and formats them.

        Parameters:
        - data (list): the data to generate batch from.

        Returns:
        - (torch.Tensor): the output tensor.
        """
        data = list(zip(*data))

        inputs = torch.cat(data[0], dim=0)
        targets = torch.cat(data[1], dim=0)

        idx = torch.randperm(inputs.shape[0])
        return inputs[idx], targets[idx]
