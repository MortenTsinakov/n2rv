"""Dense layer aka fully connected layer."""

import numpy as np
from n2rv.activations.activation import Activation
from n2rv.layers.layer import Layer
from n2rv.weight_initialization import weight_initialization


class Dense(Layer):
    """Fully connected layer class."""

    def __init__(
        self,
        output_size: int = None,
        activation: Activation = None,
        weights_initializer: str = None,
        bias_initializer: str = "zeros",
        name: str = "Dense",
    ) -> None:
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
        """
        On successful validation attach this layer onto the previous
        layer, calculate the input size using the previous layer and
        create the initial weights.

        Inputs:
            previous_layer (Layer) - the layer in the network before this one.
        Return:
            Layer - this layer
        """
        self.validate_call(previous_layer)
        self.previous_layer = previous_layer
        if isinstance(previous_layer.output_size, int):
            self.input_size = previous_layer.output_size
        else:
            self.input_size = np.prod(list(previous_layer.output_size))
        self.create_weights()
        return self

    def create_weights(self):
        """
        Initialize weights and biases for the layer according to the weight
        initialization technique defined on creation of the layer.
        """
        self.weights = weight_initialization.get_weights(
            self.input_size, self.output_size, self.weights_initializer
        )
        self.bias = weight_initialization.get_weights(
            1, self.output_size, self.bias_initializer
        )

    def forward(self, input_data: np.ndarray):
        """
        Calculate the output of the layer during forward pass.

        Inputs:
            input_data (np.ndarray) - the data coming from the previous layer.
        Return:
            np.ndarray - (matrix) multiply the incoming data with the layer
                         weights, add bias and pass the result through the
                         activation function.
        """
        self.input = np.reshape(input_data, (input_data.shape[0], -1))
        self.output = np.dot(self.input, self.weights) + self.bias
        self.output = self.activation.forward(self.output)
        return self.output

    def backward(self, output_error: np.ndarray) -> np.ndarray:
        """
        Calculate the gradients for weights, biases and the error
        for previous layer.

        Inputs:
            output_error (np.ndarray) - the error in the output
        Return:
            np.ndarray - the error in the input
        """
        input_error = np.dot(output_error, self.weights.T)
        output_error = self.activation.backward(output_error)
        self.grad_weights = np.dot(self.input.T, output_error)
        self.grad_bias = output_error.sum(axis=0)
        return input_error

    def update(self,
               weights_change: np.ndarray,
               bias_change: np.ndarray) -> None:
        """
        Update the weights and biases by the amount calculated by
        the optimizer of the model.

        Inputs:
            weight_change (np.ndarray) - the amounts that should be
                subtracted from the weight matrix
            bias_change (np.ndarray) - the amount that should be
                subtracted from the bias matrix
        """
        self.weights -= weights_change
        self.bias -= bias_change

    def validate_init(self, output_size: int, name: str):
        """
        Validate parameters provided during initialization.

        Inputs:
            output_size (int) - the size of the layer outputs
            name (str) - the name of the layer
        """
        if not isinstance(output_size, int):
            raise ValueError(
                "Output size for Dense layer has to be an "
                + f"integer. Got: {type(output_size)}"
            )
        if output_size <= 0:
            raise ValueError(
                "Output size for Dense layer has to be a "
                + f"positive integer. Got: {output_size}"
            )
        if not isinstance(name, str):
            raise ValueError(
                "Name for Dense layer has to be of string " +
                f"type. Got: {type(name)}"
            )

    def validate_call(self, previous_layer):
        """Validate parameters provided during calling the layer."""
        if not isinstance(previous_layer, Layer):
            raise ValueError(
                "The preceding layer for Dense layer has to "
                + f"be of type Layer. Got {type(previous_layer)}"
            )
