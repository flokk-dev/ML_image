"""
Creator: HOCQUET Florian, LANDZA Houdi
Date: 30/09/2022
Version: 1.0

Purpose: Trains a given model.
"""

# IMPORT: utils
import os
import csv
import time
from tqdm import tqdm

# IMPORT: deep learning
import torch
import torch.nn as nn
from torch.nn import L1Loss, MSELoss, HuberLoss

# IMPORT: fine tunning
from ray import tune

# IMPORT: project
import paths
from src.metrics import SAE, RMSE


class Trainer:
    """
    The class training a given model.

    Attributes:
    - _model (Model): the model to train.
    - _name (str): the training's name.
    - _parameters (dict): the training's parameters.
    - _tuning (bool): if True, tune the model.
    - _loader: the loading manager.
    - _loaders: the training and validation DataLoaders.
    - _loss: the loss to minimize.
    - _optimizer: the training's optimizer.
    - _scheduler: the scheduler managing the learning rate's evolution.
    - _checkpoint: the training's checkpoint.
    - _loss_results: the training and validation losses' results.
    - _visualizer (Visualiser): the training's visualizer.

    Methods:
    - launch_train: Launch the training.
    - _train: Trains the model.
    - _train_fn: The training function.
    - _train_block: The training block.
    - _save: Saves the model's weights, its parameters and its results.
    - _save_results: Saves the model's parameters and its results.
    - _tune: Tunes the model's hyperparameters.
    """
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, params: dict):
        """
        Initialize the Trainer class.

        Parameters:
        - params (dict): the training's parameters.
        - tuning (bool): if True, tune the model.
        """
        # Model and data
        self._model = None
        self._name = None

        self._params = params

        self._loader = None
        self._loaders = {"train": None, "valid": None}

        # Parameters
        self._optimizer, self._scheduler = None, None

        self._loss = HuberLoss()
        self._metrics = {"SAE": SAE(), "MAE": L1Loss(), "MSE": MSELoss(), "RMSE": RMSE()}

        # Results
        self._checkpoint = {"loss": 1000000, "epoch": 0, "duration": 0, "weights": dict()}

        self._loss_results = {"train": list(), "valid": list()}
        self._metrics_results = {key: {"train": list(), "valid": list()} for key in self._metrics}

        # Visualization
        self._visualizer = None

    def launch(self):
        """
        Initialize the training.
        """
        # Clean cache
        torch.cuda.empty_cache()

        # Parallelize GPU usage
        if torch.cuda.is_available():
            self._model = nn.DataParallel(self._model).to(torch.device(self._DEVICE))
            self._loss = self._loss.to(torch.device(self._DEVICE))

            for key in self._metrics.keys():
                self._metrics[key].to(torch.device(self._DEVICE))

        # Generate the loaders
        self._loaders["train"], self._loaders["valid"] = self._loader.get_data_loader()

        # Launch Training
        self._train()

    def _train(self):
        """
        Launch the training.
        """
        print("\nLancement de l'entrainement.")

        time.sleep(1)
        for epoch_idx in tqdm(range(self._params["epochs"])):
            # Clear cache
            torch.cuda.empty_cache()

            # Training and validation
            self._train_epoch(step="train")
            self._train_epoch(step="valid")

            # Plot results
            self._visualizer.plot_res(self._loss_results, self._metrics_results, epoch_idx,
                                      self._scheduler.get_last_lr()[0])

            # Update checkpoint
            if self._loss_results["valid"][epoch_idx] < self._checkpoint["loss"]:
                self._checkpoint["loss"] = self._loss_results["valid"][epoch_idx]
                self._checkpoint["epoch"] = epoch_idx
                self._checkpoint["duration"] = 0
                self._checkpoint["weights"] = self._model.module.state_dict()
            else:
                self._checkpoint["duration"] += 1

            # Verify checkpoint's age
            if self._checkpoint["duration"] >= 5:
                break

            self._scheduler.step()

        # Save the results and end the visualization
        self._save()
        time.sleep(60)

        self._visualizer.finish_run()

    def _train_epoch(self, step: str):
        """
        Trains the model.

        Parameters:
        - step (str): the current training's step.
        """
        # Training
        epoch_loss = list()
        epoch_metric = dict()
        for key in self._metrics:
            epoch_metric[key] = list()
            epoch_metric[f"{key}_inputs"] = list()

        batch_idx = 0
        if self._params["lazy_loading"]:
            for batch in self._loaders[step]:
                for i in range(0, batch[0].shape[0], self._params["batch_size"]):
                    batch_loss, batch_metric = self._train_block(
                        inputs=batch[0][i: i + self._params["batch_size"]],
                        targets=batch[1][i: i + self._params["batch_size"]],
                        batch_idx=batch_idx,
                        step=step
                    )

                    epoch_loss.append(batch_loss)
                    for key in epoch_metric.keys():
                        epoch_metric[key].append(batch_metric)

                    batch_idx += 1
        else:
            for batch in self._loaders[step]:
                batch_loss, batch_metric = self._train_block(
                    inputs=batch[0],
                    targets=batch[1],
                    batch_idx=batch_idx,
                    step=step
                )

                epoch_loss.append(batch_loss)
                for key in epoch_metric:
                    epoch_metric[key].append(batch_metric[key])

                batch_idx += 1

        # Results
        current_loss = sum(epoch_loss) / len(epoch_loss)
        self._loss_results[step].append(current_loss)

        for key in epoch_metric.keys():
            if key not in self._metrics_results:
                self._metrics_results[key] = {"train": list(), "valid": list()}
            self._metrics_results[key][step].append(sum(epoch_metric[key]) / len(epoch_metric[key]))

    def _train_block(self, inputs: torch.Tensor, targets: torch.Tensor, batch_idx: int, step: str):
        """
        The training's function.

        Parameters:
        - inputs (str): the inputs tensor.
        - targets (str): the targets tensor.
        - batch_idx (int): the current batch's index.
        - step (str): the current training's step.
        """
        # Clear GPU cache
        torch.cuda.empty_cache()

        # INPUTS -> (batch_size, 5, 512, 512)
        inputs = inputs.type(torch.float32).to(torch.device(self._DEVICE))

        # TARGETS -> (batch_size, 1, 512, 512)
        targets = targets.type(torch.float32).to(torch.device(self._DEVICE))

        self._optimizer.zero_grad()
        with torch.set_grad_enabled(step == "train"):
            # Calculate the results
            logits = self._model(inputs)
            batch_loss = self._loss(logits, targets)

            if step == "train":
                batch_loss.backward()
                self._optimizer.step()

        # LOSS AND METRIC
        loss_value = batch_loss.item()
        metric_value = dict()
        for key in self._metrics:
            metric_value[key] = self._metrics[key](logits, targets).item()
            metric_value[f"{key}_inputs"] = self._metrics[key](logits, inputs).item()

        if batch_idx % 50 == 0:
            self._visualizer.plot_images(inputs, targets, logits, step=step)
        return loss_value, metric_value

    def _save(self):
        """
        Saves the model's weights, its parameters and its results.
        """
        save_path = os.path.join(paths.MODEL_SAVE_PATH, f"{self._name}.pt")
        torch.save(self._checkpoint["weights"], save_path)

        self._save_results()
        print(f"epoch: {self._checkpoint['epoch']}, "
              f"duration: {self._checkpoint['duration']}, "
              f"loss_value: {self._checkpoint['loss']}")

    def _save_results(self):
        """
        Saves the model's parameters and its results.
        """
        info = dict()

        info["model_name"] = self._name
        info["data_path"] = self._loader.path
        info["loss_function"] = self._loss.__str__()

        for key, value in self._params.items():
            info[key] = value

        info["train_loss"] = self._loss_results["train"]
        info["valid_loss"] = self._loss_results["valid"]

        for metric, values in self._metrics_results.items():
            info[f"train_{metric}"] = values["train"]
            info[f"valid_{metric}"] = values["valid"]

        if not os.path.exists(paths.TRAIN_RES_PATH):
            with open(paths.TRAIN_RES_PATH, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(list(info.keys()))

        with open(paths.TRAIN_RES_PATH, "a", newline="", encoding="utf-8") as f:
            dictwriter_object = csv.DictWriter(f, fieldnames=list(info.keys()))
            dictwriter_object.writerow(info)
            f.close()

    def __str__(self):
        return f"{self._model.name}(" + \
               f"nb_data={self._params['nb_data']}, " + \
               f"workers={self._params['workers']}, " + \
               f"epochs={self._params['epochs']}, " + \
               f"lr={self._params['lr']}, " + \
               f"batch_size={self._params['batch_size']}," + \
               f"lr_multiplier={self._params['lr_multiplier']})"
