from layer import Layer
from activation import Activation
import numpy as np


class Linear(Layer):
    def __init__(self, input_size, output_size, activation=None) -> None:
        # super().__init__()
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5
        self.activation = Activation(activation)

    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.activation.forward(self.output)

    def backward(self, output_error, learning_rate):
        output_error = self.activation.backward(output_error, learning_rate)
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        self.weights += -learning_rate * weights_error
        self.bias += -learning_rate * output_error
        return input_error
