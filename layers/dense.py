from layers.layer import Layer
from activations.activation import Activation
from weight_initialization import weight_initialization
import numpy as np


class Dense(Layer):
    def __init__(self,
                 output_size=None,
                 activation=None,
                 weights_initializer=None,
                 bias_initializer='zeros',
                 name="Dense") -> None:
        self.validate_init(output_size, name)
        self.output_size = output_size
        self.activation = Activation(activation)
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.previous_layer = None
        self.name = name

    def __call__(self, previous_layer):
        self.validate_call(previous_layer)
        self.previous_layer = previous_layer
        if (isinstance(previous_layer.output_size, int)):
            self.input_size = previous_layer.output_size
        else:
            self.input_size = np.prod(list(previous_layer.output_size))
        self.create_weights()
        return self

    def create_weights(self):
        self.weights = weight_initialization.get_weights(
            self.input_size,
            self.output_size,
            self.weights_initializer
        )
        self.bias = weight_initialization.get_weights(
            1,
            self.output_size,
            self.bias_initializer
        )

    def forward(self, input_data):
        self.input = np.reshape(input_data, (input_data.shape[0], -1))
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

    def validate_init(self, output_size, name):
        if not isinstance(output_size, int):
            raise ValueError("Output size for Dense layer has to be an " +
                             f"integer. Got: {type(output_size)}")
        if output_size <= 0:
            raise ValueError("Output size for Dense layer has to be a " +
                             f"positive integer. Got: {output_size}")
        if not isinstance(name, str):
            raise ValueError("Name for Dense layer has to be of string " +
                             f"type. Got: {type(name)}")

    def validate_call(self, previous_layer):
        if not isinstance(previous_layer, Layer):
            raise ValueError("The preceding layer for Dense layer has to " +
                             f"be of type Layer. Got {type(previous_layer)}")
