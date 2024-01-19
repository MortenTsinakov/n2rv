from layers.layer import Layer
from exceptions import exception


class Input(Layer):
    def __init__(self, shape) -> None:
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

    def forward(self, inputs):
        self.output = inputs
        return self.output

    def backward(self, output_error):
        return output_error

    def update(self):
        pass

    def validate_init(self, shape):
        if not isinstance(shape, tuple):
            raise TypeError("Input shape should be of type " +
                            f"tuple. Got: {type(shape)}")
        if len(shape) == 0:
            raise ValueError("Input shape should not be empty.")
        for element in shape:
            if element <= 0:
                raise ValueError("Input shape should consist only " +
                                 f"of positive integers. Got {shape}")
