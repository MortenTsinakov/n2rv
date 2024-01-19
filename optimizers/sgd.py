"""Stochastic Gradient Descent optimizer."""


from optimizers.optimizer import Optimizer
from layers.layer import Layer


class SGD(Optimizer):
    """SGD class."""
    def __init__(self, learning_rate: float = 0.01) -> None:
        self.validata_correct_lr_arg(learning_rate)
        self.learning_rate = learning_rate

    def update(self, layers: list[Layer]) -> None:
        """Update the parameters of all the layers in the network."""
        for layer in layers:
            if layer.trainable:
                weights_change = self.learning_rate * layer.grad_weights
                bias_change = self.learning_rate * layer.grad_bias
                layer.update(weights_change, bias_change)

    def validata_correct_lr_arg(self, learning_rate: float) -> None:
        """Make sure that the learning rate is of correct type and positive."""
        if not isinstance(learning_rate, float):
            raise TypeError(
                "Learning rate argument in Model.fit() " +
                f"should be a float. Got: {type(learning_rate)}")
        if learning_rate <= 0:
            raise ValueError(
                "Learning rate argument in Model.fit() " +
                f"should be >0. Got: {learning_rate}"
            )