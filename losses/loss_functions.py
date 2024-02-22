"""All available loss functions."""


import numpy as np


def mse(y_true: np.ndarray,
        y_pred: np.ndarray) -> float:
    """Mean Squared Error loss function."""
    return np.mean(np.power(y_true - y_pred, 2))


def mse_derivative(y_true: np.ndarray,
                   y_pred: np.ndarray) -> np.ndarray:
    """Derivative function for Mean Squared Error loss."""
    return 2 * (y_pred - y_true) / y_true.size


def categorical_cross_entropy_with_softmax(y_true: np.ndarray,
                                           y_pred: np.ndarray) -> np.ndarray:
    """Categorical Cross Entropy loss function."""
    epsilon = 1e-15
    return -np.mean(y_true * np.log(y_pred + epsilon))


def categorical_cross_entropy_with_softmax_derivative(y_true: np.ndarray,
                                                      y_pred: np.ndarray) -> np.ndarray:
    """Derivative function for categorical cross entropy."""
    return y_pred - y_true


def binary_cross_entropy(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         epsilon: float =1e-7) -> np.ndarray:
    """Binary Cross Entropy loss function"""
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return np.mean(-(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))


def binary_cross_entropy_derivative(y_true: np.ndarray,
                                    y_pred: np.ndarray,
                                    epsilon: float = 1e-7) -> float:
    """Derivative function for Binary Cross Entropy loss"""
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return (y_pred - y_true) / (y_pred * (1 - y_pred))


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
    'binary_cross_entropy':
        [binary_cross_entropy,
         binary_cross_entropy_derivative]
}
