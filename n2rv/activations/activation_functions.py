"""All available activation functions."""

import numpy as np


def tanh(x: np.ndarray) -> np.ndarray:
    """
    Tanh (Hyperbolic tangent) activation function.

    Inputs:
        x (np.ndarray) - the input values
    Return:
        np.ndarray - x with tanh applied
    """
    return np.tanh(x)


def tanh_derivative(x: np.ndarray, output_error: np.ndarray) -> np.ndarray:
    """
    Derivative of tanh function.

    Inputs:
        x (np.ndarray) - the input values
        output_error (np.ndarray) - the error in the output
    Return:
        np.ndarray - the derivative of the tanh function evaluated at x
                     and multiplied with output error.
    """
    return (1 - np.tanh(x) ** 2) * output_error


def relu(x: np.ndarray) -> np.ndarray:
    """
    ReLU (Rectified Linear Unit) activation function.

    Inputs:
        x (np.ndarray) - the input values
    Return:
        np.ndarray - x with ReLU applied
    """
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray, output_error: np.ndarray) -> np.ndarray:
    """
    Derivative of ReLU function.

    Inputs:
        x (np.ndarray) - the input values
        output_error (np.ndarray) - the error in the output
    Return:
        np.ndarray - the derivative of the ReLU function evaluated at x
                     and multiplied with output error.
    """
    return np.where(x >= 0, 1, 0).astype(np.int8) * output_error


def leaky_relu(x: np.ndarray) -> np.ndarray:
    """
    Leaky ReLU activation function.

    Inputs:
        x (np.ndarray) - the input values
    Return:
        np.ndarray - x with Leaky ReLU applied
    """
    return np.maximum(0.01 * x, x)


def leaky_relu_derivative(x: np.ndarray,
                          output_error: np.ndarray) -> np.ndarray:
    """
    Derivative of Leaky ReLU function.

    Inputs:
        x (np.ndarray) - the input values
        output_error (np.ndarray) - the error in the output
    Return:
        np.ndarray - the derivative of the Leaky ReLU function evaluated at x
                     and multiplied with output error.
    """
    return np.where(x >= 0, 1, 0.01) * output_error


def sigmoid(x: np.ndarray):
    """
    Sigmoid activation function.

    Inputs:
        x (np.ndarray) - the input values
    Return:
        np.ndarray - x with Sigmoid function applied
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: np.ndarray, output_error: np.ndarray) -> np.ndarray:
    """
    Derivative of sigmoid function.

    Inputs:
        x (np.ndarray) - the input values
        output_error (np.ndarray) - the error in the output
    Return:
        np.ndarray - the derivative of the Sigmoid function evaluated at x
                     and multiplied with output error.
    """
    sig_x = sigmoid(x)
    return (sig_x * (1 - sig_x)) * output_error


def linear(x: np.ndarray) -> np.ndarray:
    """
    Linear activation function.

    Inputs:
        x (np.ndarray) - the input values
    Return:
        x (np.ndarray)
    """
    return x


def linear_derivative(x: np.ndarray, output_error: np.ndarray) -> np.ndarray:
    """
    Derivative of linear function.

    Inputs:
        x (np.ndarray) - the input values
        output_error (np.ndarray) - the error in the output
    Return:
        output_error (np.ndarray) - the derivative of the linear
                                    function is just the output error.
    """
    return output_error


def softmax_with_categorical_cross_entropy(x: np.ndarray) -> np.ndarray:
    """
    Stable softmax function.

    Inputs:
        x (np.ndarray) - the input values
    Return:
        np.ndarray - x with Softmax function applied
    """
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    exps = np.exp(x_shifted)
    return exps / np.sum(exps, axis=1, keepdims=True)


def softmax_with_categorical_cross_entropy_derivative(
    x: np.ndarray, output_error: np.ndarray
) -> np.ndarray:
    """
    Softmax activation derivative function (combined with
    Categorical Cross-Entropy).

    Inputs:
        x (np.ndarray) - the input values
        output_error (np.ndarray) - the error in the output
    Return:
        output_error (np.ndarray) - the derivative is calculated
            by the Categorical-Cross Entropy derivative function
            and this function just passes it on.
    """
    return output_error


def get_function(function):
    """
    Return the list of [activation function, derivative
    of activating function] by string.
    """
    if function not in activation_functions:
        raise ValueError(
            "Requested activation function doesn't" + f"excist: {function}"
        )
    return activation_functions[function]


# Dictionary of activation functions and their derivatives.
activation_functions = {
    "tanh": [tanh, tanh_derivative],
    "relu": [relu, relu_derivative],
    "leaky_relu": [leaky_relu, leaky_relu_derivative],
    "sigmoid": [sigmoid, sigmoid_derivative],
    "linear": [linear, linear_derivative],
    "softmax": [
        softmax_with_categorical_cross_entropy,
        softmax_with_categorical_cross_entropy_derivative,
    ],
}
