"""Base layer for all the different layers available."""
from numpy import ndarray


class Layer:
    """Base layer class for other layers."""
    def __init__(self) -> None:
        self.input = None
        self.output = None

    def forward(self, input_data: ndarray) -> ndarray:
        """Calculate values on forward pass through the layer."""
        raise NotImplementedError

    def backward(self, output_error: ndarray) -> ndarray:
        """Calculate gradients for weights and biases and error for the previous layer in the network."""
        raise NotImplementedError

    def update(self, weights_change: ndarray, bias_change: ndarray) -> ndarray:
        """Update weights and biases according to the values calculated by the model optimizer."""
        raise NotImplementedError
