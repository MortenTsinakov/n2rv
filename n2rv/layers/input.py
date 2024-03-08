"""Input layer - the first layer in the network."""

import numpy as np
from n2rv.exceptions import exception
from n2rv.layers.layer import Layer


class Input(Layer):
    """Input layer class."""

    def __init__(self, shape: tuple) -> None:
        super().__init__()
        self.validate_init(shape)
        self.shape = shape
        self.output_size = shape
        self.previous_layer = None
        self.trainable = False

    def __call__(self):
        raise exception.IncompatibleLayerError(
            "Input layer can't be used mid-model. Use it only "
            + "as the first layer in your model."
        )

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Pass the inputs through the layer.

        Inputs:
            input_data (np.ndarray) - incoming data
        Return:
            self.output (np.ndarray) - inputs and outputs for this layer
                are exactly the same
        """
        self.output = input_data
        return self.output

    def backward(self, output_error: np.ndarray) -> np.ndarray:
        """
        Pass the output error through the layer.

        Inputs:
            output_error (np.ndarray) - the error in the output
        Return:
            output_error (np.ndarray) - no calculations take place in Input
                layer, just pass on the error coming in
        """
        return output_error

    def update(self,
               weights_change: np.ndarray,
               bias_change: np.ndarray) -> None:
        """
        Do nothing as the Input layer has no weights or biases to update.

        Inputs:
            weights_change (np.ndarray)
            bias_change (np.ndarray)
        """
        pass

    def validate_init(self, shape):
        """
        Validate the parameters provided when initializing the layer.

        Inputs:
            shape (tuple) - The shape of the incoming data.
        """
        if not isinstance(shape, tuple):
            raise TypeError(
                "Input shape should be of type " + f"tuple. Got: {type(shape)}"
            )
        if len(shape) == 0:
            raise ValueError("Input shape should not be empty.")
        for element in shape:
            if element <= 0:
                raise ValueError(
                    "Input shape should consist only "
                    + f"of positive integers. Got {shape}"
                )
