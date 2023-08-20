import numpy as np


def mse(y_true, y_pred):
    """Mean squared error loss function."""
    return np.mean(np.power(y_true - y_pred, 2))


def mse_derivative(y_true, y_pred):
    """Derivative function for mean squared error loss."""
    return 2 * (y_pred - y_true) / y_true.size


def cat_cross_e(y_true, y_pred):
    """Categorical cross entropy loss function."""
    return -np.sum(y_true * np.log(y_pred))


def cat_cross_e_derivative(y_true, y_pred):
    """Derivative function for categorical cross entropy."""
    return -y_true / y_pred


def get_loss(function):
    if function not in losses.keys():
        raise ValueError(
            f"Requested loss is not supported: {function}"
        )
    return losses[function]


losses = {
    'mse': [mse, mse_derivative],
    'categorical_cross_entropy': [cat_cross_e, cat_cross_e_derivative],
}
