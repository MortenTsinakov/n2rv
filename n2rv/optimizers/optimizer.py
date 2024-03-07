"""Optimizer base class for different optimizers."""

from n2rv.layers.layer import Layer


class Optimizer:
    """Optimizer base class."""

    def update(self, layers: list[Layer]) -> None:
        """
        Update the weights and biases of all the layers in the network.

        Inputs:
            layers (list[Layer]) - list of layers in the network
        """
        raise NotImplementedError
