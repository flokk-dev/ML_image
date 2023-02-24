"""
Creator: HOCQUET Florian, Landza HOUDI
Date: 30/09/2022
Version: 1.0

Purpose: Manages the project.
"""

# IMPORT: utils
import argparse

# IMPORT: deep learning
import torch

# IMPORT: projet
from src import Trainer2D, Trainer25D, Trainer3D, Inference2D, Inference25D, Inference3D

# WARNINGS SHUT DOWN
import warnings
warnings.filterwarnings("ignore")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Get model training parameters.")

    parser.add_argument("-s", "--src", type=str, nargs="?", help="the dataset's path.")

    parser.add_argument("-m", "--model", type=str, nargs="?", default=None,
                        help="the model's path.")

    parser.add_argument("-p", "--pipe", type=str, nargs="?",
                        choices=["train", "inference"], default="train",
                        help="the type of pipeline to use")

    parser.add_argument("-ll", "--lazy_loading", type=str, nargs="?", default=False,
                        help="the loading method.")

    parser.add_argument("-n", "--nb_data", type=int, nargs="?", default=1000,
                        help="the number of data to train on.")

    parser.add_argument("-d", "--dim", type=str, nargs="?", default="2D",
                        help="the model's data dimension.")

    parser.add_argument("-ph", "--patch_height", type=int, nargs="?", default=5,
                        help="the height of the patches in 2.5D.")

    return parser


def get_params(args) -> dict:
    params = {
        "lazy_loading": args.lazy_loading,
        "nb_data": args.nb_data,
        "patch_height": args.patch_height if args.dim == "25D" else None,
        "batch_size": 32,
        "epochs": 25,
        "workers": 1,
        "lr": 1e-2,
        "lr_multiplier": 0.9,
        "valid_coeff": 0.2
    }

    return params


if __name__ == "__main__":
    # Free the memory
    torch.cuda.empty_cache()

    # Parameters
    parser = get_parser()
    args = parser.parse_args()
    params = get_params(parser.parse_args())

    print()
    for key, value in params.items():
        print(f"{key}: {value}")

    # Training
    tasks = {
        "train": {"2D": Trainer2D, "25D": Trainer25D, "3D": Trainer3D},
        "inference": {"2D": Inference2D, "25D": Inference25D, "3D": Inference3D}
    }

    task = tasks[args.pipe][args.dim](data_path=args.src, params=params, model_path=args.model)
    task.launch()
