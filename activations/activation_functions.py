import numpy as np


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x, output_error):
    return (1 - np.tanh(x) ** 2) * output_error


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x, output_error):
    return ((x >= 0).view('i1')) * output_error


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x, output_error):
    sig_x = sigmoid(x)
    return (sig_x * (1 - sig_x)) * output_error


def softmax(x):
    """Stable softmax function."""
    x_shifted = x - np.max(x)
    exps = np.exp(x_shifted)
    return exps / np.sum(exps)


def softmax_derivative(x, output_error):
    s = softmax(x)
    ds = np.diag(s) - np.outer(s, s.T)
    dx = np.dot(output_error, ds)
    return dx


def get_function(function):
    if function not in activation_functions.keys():
        raise ValueError(
            f"Requested activation function doesn't excist: {function}"
            )
    return activation_functions[function]


activation_functions = {
    'tanh': [tanh, tanh_derivative],
    'relu': [relu, relu_derivative],
    'sigmoid': [sigmoid, sigmoid_derivative],
    'softmax': [softmax, softmax_derivative],
}
