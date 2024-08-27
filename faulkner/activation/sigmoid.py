import numpy as np


def sigmoid(x: np.ndarray):
    """
    Apply the sigmoid activation function element-wise to the input array.

    Parameters:
    x (numpy.ndarray or float): Input array or scalar.

    Returns:
    numpy.ndarray or float: Output array or scalar with sigmoid applied.
    """
    return 1 / (1 + np.exp(-x))
