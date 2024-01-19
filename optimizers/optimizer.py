"""Optimizer base class for different optimizers."""
from layers.layer import Layer


class Optimizer:
    """Optimizer base class."""
    def __init__(self) -> None:
        raise NotImplementedError

    def update(self, layers: list[Layer]) -> None:
        """Update the weights and biases of all the layers in the network."""
        raise NotImplementedError
