"""
Creator: HOCQUET Florian, LANDZA Houdi
Date: 30/09/2022
Version: 1.0

Purpose: Manages the project's utils methods.
"""

# IMPORT: utils
import random
import datetime

# IMPORT: data process
import numpy as np

# IMPORT: data visualization
import matplotlib.pyplot as plt

# IMPORT: deep learning
import torch


"""""
BASIC UTILS
"""""


def get_datetime():
    """
    Returns the current date.
    """
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def get_max_item(dictionary):
    """
    Returns the current date.
    """
    key = max(dictionary, key=dictionary.get)
    return key, dictionary[key]


"""""
TENSOR PROCESS
"""""


def numpy_to_tensor(path):
    """
    Convert a numpy.ndarray into a Tensor.

    Parameters:
    - path (str) path to the npy file containing the numpy.ndarray.
    """
    return torch.from_numpy(np.load(path).astype(np.int16)).type(torch.float32)


def resample(tensor, shape):
    """
    Resample a tensor to the desired shape.

    Parameters:
    - tensor (torch.Tensor): the tensor to resample.
    - shape (int): the final shape desired.
    """
    if tensor.shape == shape:
        return tensor

    tensor = torch.nn.functional.interpolate(add_dim(tensor, nb_dim=2), size=shape)
    return remove_dim(tensor, nb_dim=2)


def pad_tensor(tensor, padding_factor=64):
    tensor_shape = tensor.shape
    new_z = padding_factor * ((tensor.shape[1] // padding_factor) + 1)

    new_tensor = torch.zeros((tensor_shape[0], new_z, tensor_shape[2], tensor_shape[3]))
    new_tensor[:, :tensor_shape[1]] = tensor

    return new_tensor


def add_dim(tensor, nb_dim=1, dim=0):
    """
    Add dimension to a tensor.

    Parameters:
    - tensor (torch.Tensor): the tensor to add dimension to.
    - nb_dim (int): the number of dimension to add.
    - dim (int): the dimension where to add dimension.
    """
    for i in range(nb_dim):
        tensor = torch.unsqueeze(tensor, dim=dim)

    return tensor


def remove_dim(tensor, nb_dim=1, dim=0):
    """
    Removes dimension to a tensor.

    Parameters:
    - tensor (torch.Tensor): the tensor to remove dimension to.
    - nb_dim (int): the number of dimension to remove.
    - dim (int): the dimension where to remove dimension.
    """
    for i in range(nb_dim):
        tensor = torch.squeeze(tensor, dim=dim)

    return tensor


def size_of(tensor: torch.Tensor) -> float:
    total = tensor.element_size()
    for shape in tensor.shape:
        total *= float(shape)

    return total / 1e6


"""""
VISUALIZATION
"""""


def get_mip(volume):
    return np.amax(volume.numpy(), axis=1)


def plot_resample(x, x_resampled, y, y_resampled, slice):
    """
    Plot resample result.

    Parameters:
    - x (torch.Tensor): the first tensor.
    - x_resampled (torch.Tensor): the first tensor resampled.
    - y (torch.Tensor): the second tensor.
    - y_resampled (torch.Tensor): the second tensor resampled.
    - slice (int): the index of the slice to plot.
    """
    fig = plt.figure(figsize=(8, 8))

    fig.add_subplot(2, 2, 1)
    plt.imshow(x[:, :, slice])
    plt.title(f"x[{slice}]\n shape({str(x.shape)[12:-2]})")

    fig.add_subplot(2, 2, 2)
    plt.imshow(x_resampled[0, :, :, slice])
    plt.title(f"x_resampled[{slice}]\n shape({str(x_resampled.shape)[12:-2]})")

    fig.add_subplot(2, 2, 3)
    plt.imshow(y[:, :, slice])
    plt.title(f"y[{slice}]\n shape({str(y.shape)[12:-2]})")

    fig.add_subplot(2, 2, 4)
    plt.imshow(y_resampled[0, :, :, slice])
    plt.title(f"y_resampled[{slice}]\n shape({str(y_resampled.shape)[12:-2]})")

    fig.tight_layout()
    plt.show()

    plt.close(fig=fig)


def save_distribution(x1, y1, x2, y2, x_label, y_label, path):
    plt.plot(x1, y1, label="target")
    plt.plot(x2, y2, label="prediction")
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.savefig(path)
    plt.close()
