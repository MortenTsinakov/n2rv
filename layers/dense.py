"""Dense layer aka fully connected layer."""


import numpy as np
from layers.layer import Layer
from activations.activation import Activation
from weight_initialization import weight_initialization


class Dense(Layer):
    """Fully connected layer class."""
    def __init__(self,
                 output_size: int         = None,
                 activation: Activation   = None,
                 weights_initializer: str = None,
                 bias_initializer: str    ='zeros',
                 name: str                ="Dense") -> None:
        super().__init__()
        self.validate_init(output_size, name)
        self.input_size = None
        self.output_size = output_size
        self.activation = Activation(activation)
        self.weights_initializer = weights_initializer
        self.weights = None
        self.grad_weights = None
        self.bias_initializer = bias_initializer
        self.bias = None
        self.grad_bias = None
        self.previous_layer = None
        self.name = name
        self.trainable = True

    def __call__(self, previous_layer: Layer) -> Layer:
        self.validate_call(previous_layer)
        self.previous_layer = previous_layer
        if isinstance(previous_layer.output_size, int):
            self.input_size = previous_layer.output_size
        else:
            self.input_size = np.prod(list(previous_layer.output_size))
        self.create_weights()
        return self

    def create_weights(self):
        """Initialize weights and biases for the layer."""
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
        """Calculate the output of the layer during forward pass."""
        self.input = np.reshape(input_data, (input_data.shape[0], -1))
        self.output = np.dot(self.input, self.weights) + self.bias
        self.output = self.activation.forward(self.output)
        return self.output

    def backward(self, output_error):
        """Calculate the gradients for weights, biases and the error for previous layer."""
        input_error = np.dot(output_error, self.weights.T)
        output_error = self.activation.backward(output_error)
        self.grad_weights = np.dot(self.input.T, output_error)
        self.grad_bias = output_error.sum(axis=0)
        return input_error

    def update(self, weights_change, bias_change):
        """Update the weights and biases by the amount calculated by the optimizer of the model."""
        self.weights -= weights_change
        self.bias -= bias_change

    def validate_init(self, output_size, name):
        """Validate parameters provided during initialization."""
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
        """Validate parameters provided during calling the layer."""
        if not isinstance(previous_layer, Layer):
            raise ValueError("The preceding layer for Dense layer has to " +
                             f"be of type Layer. Got {type(previous_layer)}")
