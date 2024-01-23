"""All available activation functions."""
import numpy as np


def tanh(x: np.ndarray) -> np.ndarray:
    """Tanh activation function."""
    return np.tanh(x)


def tanh_derivative(x: np.ndarray, output_error: np.ndarray) -> np.ndarray:
    """Derivative of tanh function."""
    return (1 - np.tanh(x) ** 2) * output_error


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation function."""
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray, output_error: np.ndarray) -> np.ndarray:
    """Derivative of ReLU function."""
    return np.where(x >= 0, 1, 0).astype(np.int8) * output_error
    # return ((x >= 0).view('i1')) * output_error


def sigmoid(x: np.ndarray):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: np.ndarray, output_error: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid function."""
    sig_x = sigmoid(x)
    return (sig_x * (1 - sig_x)) * output_error


def linear(x: np.ndarray) -> np.ndarray:
    """Linear activation function."""
    return x


def linear_derivative(x: np.ndarray, output_error: np.ndarray) -> np.ndarray:
    """Derivative of linear function."""
    return output_error


def softmax_with_categorical_cross_entropy(x: np.ndarray) -> np.ndarray:
    """Stable softmax function."""
    x_shifted = x - np.max(x)
    exps = np.exp(x_shifted)
    return exps / np.sum(exps)


def softmax_with_categorical_cross_entropy_derivative(x: np.ndarray, output_error: np.ndarray) -> np.ndarray:
    """Softmax activation derivative function (combined with Categorical Cross-Entropy)."""
    return output_error


def get_function(function):
    """Return the list of [activation function, derivative of activating function] by string."""
    if function not in activation_functions:
        raise ValueError(
            f"Requested activation function doesn't excist: {function}"
            )
    return activation_functions[function]


# Dictionary of activation functions and their derivatives.
activation_functions = {
    'tanh': [tanh, tanh_derivative],
    'relu': [relu, relu_derivative],
    'sigmoid': [sigmoid, sigmoid_derivative],
    'linear': [linear, linear_derivative],
    'softmax': [softmax_with_categorical_cross_entropy,
                softmax_with_categorical_cross_entropy_derivative],
}
