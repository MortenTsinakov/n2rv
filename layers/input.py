from layers.layer import Layer
from exceptions import exception


class Input(Layer):
    def __init__(self, shape) -> None:
        self.shape = shape
        self.output_size = shape
        self.previous_layer = None

    def __call__(self, *args, **kwargs):
        raise exception.IncompatibleLayerError(
            "Input layer can't be used mid-model. Use it only " +
            "as the first layer in your model."
        )

    def forward(self, inputs):
        self.output = inputs
        return self.output

    def backward(self, output_error, learning_rate):
        return output_error

    def update(self, learning_rate):
        pass
