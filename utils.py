"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import sys
import datetime

# IMPORT: dataset processing
import torch


def get_datetime():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def get_max_item(dictionary):
    key = max(dictionary, key=dictionary.get)
    return key, dictionary[key]


def size_of(tensor: torch.Tensor) -> float:
    total = tensor.element_size()
    for shape in tensor.shape:
        total *= float(shape)

    return total / 1e6


def str_to_class(class_name):
    return getattr(sys.modules[__name__], class_name)


def parse_shape(input_shape, target_shape):
    return {
        "spatial_dims": len(input_shape) - 2,
        "img_size": tuple(input_shape[2:]),
        "in_channels": input_shape[1],
        "out_channels": target_shape[1]
    }
