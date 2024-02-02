"""Stochastic Gradient Descent optimizer."""


from optimizers.optimizer import Optimizer
from layers.layer import Layer


class SGD(Optimizer):
    """SGD class."""
    def __init__(self,
                 learning_rate: float = 0.01) -> None:
        self.validate_correct_lr_arg(learning_rate)
        self.learning_rate = learning_rate

    def update(self, layers: list[Layer]) -> None:
        """Update the parameters of all the layers in the network."""
        for layer in layers:
            if layer.trainable:
                layer.update(self.learning_rate * layer.grad_weights,
                             self.learning_rate * layer.grad_bias)

    def validate_correct_lr_arg(self, learning_rate: float) -> None:
        """Make sure that the learning rate is of correct type and positive."""
        if not isinstance(learning_rate, float):
            raise TypeError(
                "Learning rate argument on SGD optimizer initialization " +
                f"should be a float. Got: {type(learning_rate)}")
        if learning_rate <= 0:
            raise ValueError(
                "Learning rate argument on SGD optimizer initialization " +
                f"should be >0. Got: {learning_rate}"
            )