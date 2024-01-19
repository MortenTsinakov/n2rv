"""Input layer - the first layer in the network."""


from layers.layer import Layer
from exceptions import exception


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
            "Input layer can't be used mid-model. Use it only " +
            "as the first layer in your model."
        )

    def forward(self, input_data):
        """Pass the inputs through the layer."""
        self.output = input_data
        return self.output

    def backward(self, output_error):
        """Pass the output error through the layer."""
        return output_error

    def update(self, weights_change, bias_change):
        """Do nothing as the Input layer has no weights or biases to update."""

    def validate_init(self, shape):
        """Validate the parameters provided when initializing the layer."""
        if not isinstance(shape, tuple):
            raise TypeError("Input shape should be of type " +
                            f"tuple. Got: {type(shape)}")
        if len(shape) == 0:
            raise ValueError("Input shape should not be empty.")
        for element in shape:
            if element <= 0:
                raise ValueError("Input shape should consist only " +
                                 f"of positive integers. Got {shape}")
