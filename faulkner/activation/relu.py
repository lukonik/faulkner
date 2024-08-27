import numpy as np


def ReLU(nd: np.ndarray) -> np.ndarray:
    """
        Apply ReLU activation function
        
        Parameters:
        x(numpy.ndarray): Input Array

        Returns:
        numpy.ndarray: Output array with ReLU applied
    """
    return np.maximum(0, nd)
