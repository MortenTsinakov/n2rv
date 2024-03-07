"""All available loss functions."""

import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Error loss function.

    Inputs:
        y_true (np.ndarray) - the actual values
        y_pred (np.ndarray) - the predicted values
    Return:
        float - the average error between the actual values
            and the predicted values.
    """
    return np.mean(np.power(y_true - y_pred, 2))


def mse_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Derivative function for Mean Squared Error loss.

    Inputs:
        y_true (np.ndarray) - the actual values
        y_pred (np.ndarray) - the predicted values
    Return:
        np.ndarray - the derivative of MSE function evaluated at
            y_pred
    """
    return 2 * (y_pred - y_true) / y_true.size


def categorical_cross_entropy_with_softmax(
    y_true: np.ndarray, y_pred: np.ndarray
) -> float:
    """
    Categorical Cross Entropy loss function.

    Inputs:
        y_true (np.ndarray) - the actual values
        y_pred (np.ndarray) - the predicted values
    Return:
        float - the average error between the true values and the
            predicted values
    """
    epsilon = 1e-15
    return -np.mean(np.sum(y_true * np.log(y_pred + epsilon), axis=-1))


def categorical_cross_entropy_with_softmax_derivative(
    y_true: np.ndarray, y_pred: np.ndarray
) -> np.ndarray:
    """
    Calculate the combined derivative of Softmax and Categorical-Cross Entropy
    function.

    Inputs:
        y_true (np.ndarray) - the actual values
        y_pred (np.ndarray) - the predicted values
    Return:
        np.ndarray - the combined derivative of Softmax and CCE function
            evaluated at y_pred
    """
    return y_pred - y_true


def binary_cross_entropy(
    y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-7
) -> float:
    """
    Binary Cross Entropy loss function

    Inputs:
        y_true (np.ndarray) - the actual values
        y_pred (np.ndarray) - the predicted values
        epsilon (float) - small value added to predictions to avoid
            log of zero.
    Return:
        float - the average error between the true values and predicted
            values.
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return np.mean(-(y_true * np.log(y_pred) +
                     (1 - y_true) * np.log(1 - y_pred)))


def binary_cross_entropy_derivative(
    y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-7
) -> np.ndarray:
    """
    Derivative function for Binary Cross Entropy loss.

    Inputs:
        y_true (np.ndarray) - the actual values
        y_pred (np.ndarray) - the predicted values
        epsilon (float) - small value added to/subtracted from predictions
            to avoid zero division.
    Return:
        np.ndarray - the derivative of BCE function evaluated
            at y_pred
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return (y_pred - y_true) / (y_pred * (1 - y_pred))


def get_loss(function: str) -> list:
    """
    Return a list of [loss function, loss function derivative] by string.

    Inputs:
        function (str) - the name of the loss function.
    Return:
        list - list of the requested function and it's derivative function.
    """
    if function not in losses:
        raise ValueError(f"Requested loss is not supported: {function}")
    return losses[function]


losses = {
    "mse": [mse, mse_derivative],
    "categorical_cross_entropy": [
        categorical_cross_entropy_with_softmax,
        categorical_cross_entropy_with_softmax_derivative,
    ],
    "binary_cross_entropy": [binary_cross_entropy,
                             binary_cross_entropy_derivative],
}
