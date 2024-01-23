"""All available loss functions."""


import numpy as np


def mse(y_true: np.ndarray,
        y_pred: np.ndarray) -> np.ndarray:
    """Mean squared error loss function."""
    return np.mean(np.power(y_true - y_pred, 2))


def mse_derivative(y_true: np.ndarray,
                   y_pred: np.ndarray) -> np.ndarray:
    """Derivative function for mean squared error loss."""
    return 2 * (y_pred - y_true) / y_true.size


def categorical_cross_entropy_with_softmax(y_true: np.ndarray,
                                           y_pred: np.ndarray) -> np.ndarray:
    """Categorical cross entropy loss function."""
    epsilon = 1e-15
    return -np.mean(y_true * np.log(y_pred + epsilon))


def categorical_cross_entropy_with_softmax_derivative(y_true: np.ndarray,
                                                      y_pred: np.ndarray) -> np.ndarray:
    """Derivative function for categorical cross entropy."""
    return y_pred - y_true


def get_loss(function: str) -> list:
    """Return a list of [loss function, loss function derivative] by string."""
    if function not in losses:
        raise ValueError(
            f"Requested loss is not supported: {function}"
        )
    return losses[function]


losses = {
    'mse': [mse, mse_derivative],
    'categorical_cross_entropy':
        [categorical_cross_entropy_with_softmax,
         categorical_cross_entropy_with_softmax_derivative],
}
