import numpy as np


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x >= 0).view('i1')


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    sig_x = sigmoid(x)
    return sig_x * (1 - sig_x)


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
}
