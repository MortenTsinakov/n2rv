"""Activation that can be attached to a layer in neural network."""
from activations import activation_functions
from numpy import ndarray


class Activation():
    """Activation class to be provided to a layer."""
    def __init__(self, activation_function: str) -> None:
        activation = activation_functions.get_function(activation_function)
        self.name = activation_function
        self.input = None
        self.output = None
        self.activation = activation[0]
        self.activation_derivative = activation[1]

    def forward(self, input_data: ndarray) -> ndarray:
        """Calculate the value of the Activation function during forward pass."""
        self.input = input_data
        self.output = self.activation(x=self.input)
        return self.output

    def backward(self, output_error: ndarray) -> ndarray:
        """Calculate the derivative of the Activation function during backward pass."""
        return self.activation_derivative(x=self.input, output_error=output_error)
