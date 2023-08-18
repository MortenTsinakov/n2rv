import numpy as np


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size


def get_loss(function):
    if function not in losses.keys():
        raise ValueError(
            f"Requested loss is not supported: {function}"
        )
    return losses[function]


losses = {
    'mse': [mse, mse_derivative],
}
