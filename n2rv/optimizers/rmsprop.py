"""RMSProp optimizer."""

import numpy as np
from n2rv.layers.layer import Layer
from n2rv.optimizers.optimizer import Optimizer


class RMSProp(Optimizer):
    """RMSProp optimizer class."""

    def __init__(self,
                 learning_rate: float = 0.001,
                 beta: float = 0.9,
                 epsilon: float = 1e-07) -> None:
        """
        Initialize RMSProp optimizer.

        Inputs:
            learning_rate (float): The learning rate.
            beta (float): The decay rate.
            epsilon (float): Small value to avoid division by zero.
        """
        self.validate_arguments(learning_rate, beta, epsilon)
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.squared_grad_avg_terms = {}

    def update(self, layers: list[Layer]) -> None:
        """
        Update the parameters for all layers in the network.

        Inputs:
            layers (list[Layer]): List of layers in the neural network.
        """
        # Initialize squared gradient average terms if not already initialized
        if not self.squared_grad_avg_terms:
            for layer in layers:
                if layer.trainable:
                    squared_grad_avg_w = np.zeros_like(layer.weights)
                    squared_grad_avg_b = np.zeros_like(layer.bias)
                    self.squared_grad_avg_terms[layer] = {
                        "w": squared_grad_avg_w,
                        "b": squared_grad_avg_b,
                    }

        # Update weights and biases for each layer
        for layer in layers:
            if layer.trainable:
                self.update_squared_grad_avg_weight_term_for_layer(layer)
                self.update_squared_grad_avg_bias_term_for_layer(layer)
                layer.update(
                    self.get_change_of_weight_for_layer(layer),
                    self.get_change_of_bias_for_layer(layer),
                )

    def update_squared_grad_avg_weight_term_for_layer(self,
                                                      layer: Layer) -> None:
        """
        Update the squared gradient average weight term for layer.

        Inputs:
            layer (Layer): The layer to update
        """
        self.squared_grad_avg_terms[layer]["w"] = (
            self.beta * self.squared_grad_avg_terms[layer]["w"]
            + (1 - self.beta) * (layer.grad_weights * layer.grad_weights)
        )

    def update_squared_grad_avg_bias_term_for_layer(self,
                                                    layer: Layer) -> None:
        """
        Update the squared gradient average bias term for layer.

        Inputs:
            layer (Layer): The layer to update
        """
        self.squared_grad_avg_terms[layer]["b"] = (
            self.beta * self.squared_grad_avg_terms[layer]["b"]
            + (1 - self.beta) * (layer.grad_bias * layer.grad_bias)
        )

    def get_change_of_weight_for_layer(self, layer: Layer) -> np.ndarray:
        """
        Calculate the change in weights for layer.

        Inputs:
            layer (Layer): The layer to update
        Returns:
            np.ndarray: The change in weights
        """
        return (
            self.learning_rate
            / np.sqrt(self.squared_grad_avg_terms[layer]["w"] + self.epsilon)
            * layer.grad_weights
        )

    def get_change_of_bias_for_layer(self, layer: Layer) -> np.ndarray:
        """
        Calculate the change in bias for layer.

        Inputs:
            layer (Layer): The layer to update
        Returns:
            np.ndarray: The change in bias
        """
        return (
            self.learning_rate
            / np.sqrt(self.squared_grad_avg_terms[layer]["b"] + self.epsilon)
            * layer.grad_bias
        )

    # VALIDATION FUNCTIONS

    def validate_arguments(
        self, learning_rate: float, beta: float, epsilon: float
    ) -> None:
        """
        Validate the initialization parameters,

        Inputs:
            learning_rate (float): learning rate set on initialization.
            beta (float): decay rate set on initialization.
            epsilon (float): small value to avoid divisioin by zero set
                on initialization.
        """
        self.validate_learning_rate(learning_rate)
        self.validate_beta(beta)
        self.validate_epsilon(epsilon)

    def validate_learning_rate(self, learning_rate: float) -> None:
        """
        Validate the learning rate argument.

        Inputs:
            learning_rate (float): learning rate set on initialization
        """
        if not isinstance(learning_rate, float):
            raise TypeError(
                "Learning rate must be of type float." +
                f"Got: {type(learning_rate)}"
            )
        if learning_rate <= 0:
            raise ValueError(
                f"Learning rate must be greater than 0. Got {learning_rate}"
            )

    def validate_beta(self, beta: float) -> None:
        """
        Validate decay rate argument beta.

        Inputs:
            beta (float): decay rate set on initialization.
        """
        if not isinstance(beta, float):
            raise TypeError(f"Beta must be of type float. Got {type(beta)}")
        if not (0 < beta < 1):
            raise ValueError(f"Beta must be in the range (0, 1). Got {beta}")

    def validate_epsilon(self, epsilon: float) -> None:
        """
        Validate epsilon argument.

        Inputs:
            epsilon (float): small number to avoid division by zero set
                on initialization.
        """
        if not isinstance(epsilon, float):
            raise TypeError("Epsilon must be of type float." +
                            f"Got {type(epsilon)}")
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be greater than 0. Got {epsilon}")
