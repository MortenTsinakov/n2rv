"""Momentum optimizer."""


import numpy as np
from optimizers.optimizer import Optimizer
from layers.layer import Layer


class Momentum(Optimizer):
    """Momentum optimizer class."""
    def __init__(self,
                 learning_rate: float = 0.01,
                 momentum: float = 0.9) -> None:
        self.validate_arguments(learning_rate, momentum)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.momentum_terms = {}

    def update(self, layers: list[Layer]) -> None:
        """Update the parameters of all layers in the network."""
        # If it's the first iteration, initialize the momentum terms for each layer.
        if not self.momentum_terms:
            for layer in layers:
                if layer.trainable:
                    weight_momentum = np.zeros_like(layer.weights)
                    bias_momentum = np.zeros_like(layer.bias)
                    self.momentum_terms[layer] = {"w": weight_momentum, 
                                                  "b": bias_momentum}
        for layer in layers:
            if layer.trainable:
                self.momentum_terms[layer]["w"] = self.momentum * self.momentum_terms[layer]["w"] \
                                                  + (1 - self.momentum) * layer.grad_weights
                self.momentum_terms[layer]["b"] = self.momentum * self.momentum_terms[layer]["b"] \
                                                  + (1 - self.momentum) * layer.grad_bias
                layer.update(self.learning_rate * self.momentum_terms[layer]["w"],
                             self.learning_rate * self.momentum_terms[layer]["b"])


    def validate_arguments(self, learning_rate, momentum) -> None:
        """Validate the initialization arguments provided by the user."""
        self.validate_correct_lr_arg(learning_rate)
        self.validate_correct_momentum_argument(momentum)

    def validate_correct_lr_arg(self, learning_rate: float) -> None:
        """Make sure that the learning rate is of correct type and positive."""
        if not isinstance(learning_rate, float):
            raise TypeError(
                "Learning rate argument on Momentum optimizer initialization " +
                f"should be a float. Got: {type(learning_rate)}")
        if learning_rate <= 0:
            raise ValueError(
                "Learning rate argument on Momentum optimizer initialization " +
                f"should be >0. Got: {learning_rate}"
            )

    def validate_correct_momentum_argument(self, momentum: float) -> None:
        """Make sure that the momentum is of correct type and in range (0, 1)."""
        if not isinstance(momentum, float):
            raise TypeError(
                "The momentum argument on Momentum optimizer initialization " +
                f"should be a float. Got {type(momentum)}"
            )
        if momentum <= 0 or momentum >= 1:
            raise ValueError(
                "The momentum argument on Momentum optimizer initialization " +
                f"should be in range (0, 1). Got: {momentum}"
            )
