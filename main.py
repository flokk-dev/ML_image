"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import argparse
import json

# import warnings
# warnings.filterwarnings("ignore")

# IMPORT: projet
import paths

from src import UnsupervisedTrainer, SupervisedTrainer


class Parser(argparse.ArgumentParser):
    def __init__(self):
        # Mother class
        super(Parser, self).__init__(description="Get models training parameters.")

        # dataset
        self.add_argument(
            "-d", "--dataset", type=str, nargs="?",
            help="path to the dataset"
        )

        # weights
        self.add_argument(
            "-w", "--weights", type=str, nargs="?",
            default=None,
            help="path to the model's weights"
        )

        # pipeline
        self.add_argument(
            "-p", "--pipeline", type=str, nargs="?",
            choices=["training", "inference"], default="training",
            help="pipeline to use"
        )

        # training type
        self.add_argument(
            "-t", "--training_type", type=str, nargs="?",
            choices=["unsupervised", "supervised"], default="supervised",
            help="the type of training"
        )

        # quantity
        self.add_argument(
            "-q", "--quantity", type=int, nargs="?",
            default=1000,
            help="quantity of data to use during the training"
        )


TASKS = {
    "training": {"unsupervised": UnsupervisedTrainer, "supervised": SupervisedTrainer},
    "inference": {"unsupervised": None, "supervised": None},
}

if __name__ == "__main__":
    # Training arguments
    parser = Parser()
    args = parser.parse_args()

    # Training parameters
    with open(paths.CONFIG_PATH) as json_file:
        training_parameters = json.load(json_file)

    # Launch training
    task = TASKS[args.pipeline][args.training_type](params=training_parameters)
    task(data_path=args.dataset, weights_path=args.weights)
