from layers.layer import Layer
from activations.activation import Activation
import numpy as np


class Linear(Layer):
    def __init__(self, input_size=None,
                 output_size=None,
                 activation=None,
                 name=None) -> None:
        # super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.activation = Activation(activation)
        self.previous_layer = None
        self.name = name
        if self.input_size and self.output_size:
            self.create_weights()

    def __call__(self, previous_layer):
        self.previous_layer = previous_layer
        self.input_size = previous_layer.output_size
        self.create_weights()
        return self

    def create_weights(self):
        self.weights = np.random.randn(self.input_size, self.output_size)
        self.bias = np.zeros(self.output_size)

    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        self.output = self.activation.forward(self.output)
        return self.output

    def backward(self, output_error):
        input_error = np.dot(output_error, self.weights.T)
        output_error = self.activation.backward(output_error)
        self.grad_weights = np.dot(self.input.T, output_error)
        self.grad_bias = output_error.sum(axis=0)
        return input_error

    def update(self, learning_rate):
        self.weights -= learning_rate * self.grad_weights
        self.bias -= learning_rate * self.grad_bias
