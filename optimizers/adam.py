"""Adam optimizer."""


import numpy as np
from optimizers.optimizer import Optimizer
from layers.layer import Layer


class Adam(Optimizer):
    """Adam optimizer class."""

    def __init__(self,
                 learning_rate: float = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-08) -> None:
        """
        Initializes an Adam optimizer instance.

        Creates an Adam optimizer with the specified hyperparameters.

        Parameters:
            learning_rate (flaot): The learning rate, controlling the step
                                   size during gradient descent.
            beta1 (float): The exponential decay rate for the first moment estimates.
            beta2 (float): The exponential decay rate for the second moment estimates.
            epsilon (float); A small value added to the denominator for numerical stability.
        """
        self.validate_arguments(learning_rate=learning_rate,
                                beta1=beta1,
                                beta2=beta2,
                                epsilon=epsilon)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.time_step = 0
        self.momentum_terms = {}
        self.squared_grad_avg_terms = {}

    def update(self, layers: list[Layer]) -> None:
        """
        Update the parameters for all layers in the network.

        Initializes the momentum terms and squared gradient average terms to zeros if they
        haven't been initialized yet.
        Iterates through the layers in the model and using helper methods updates the
        layer weights and biases.
        
        Parameters:
            layers (list[Layer]): List of layers in the neural network.
        """
        # Initialize momentum terms and squared gradient average terms before training begins
        if self.time_step == 0:
            for layer in layers:
                if layer.trainable:
                    momentum_term_w = np.zeros_like(layer.weights)
                    momentum_term_b = np.zeros_like(layer.bias)
                    squared_grad_avg_w = np.zeros_like(layer.weights)
                    squared_grad_avg_b = np.zeros_like(layer.bias)
                    self.momentum_terms[layer] = {"w": momentum_term_w,
                                                  "b": momentum_term_b}
                    self.squared_grad_avg_terms[layer] = {"w": squared_grad_avg_w,
                                                          "b": squared_grad_avg_b}
        # Iterate through all the layers and update the parameters for trainable ones
        self.time_step += 1
        for layer in layers:
            if layer.trainable:
                self.update_momentum_weight_term_for_layer(layer)
                self.update_momentum_bias_term_for_layer(layer)
                self.update_squared_grad_avg_weight_term_for_layer(layer)
                self.update_squared_grad_avg_bias_term_for_layer(layer)
                # Momentum term for weights with bias correction applied
                m_hat_w = self.get_momentum_weight_term_with_corrected_bias_for_layer(layer)
                # Momentum term for biases with bias correction applied
                m_hat_b = self.get_momentum_bias_term_with_corrected_bias_for_layer(layer)
                # Squared gradient average term for weights with bias correction applied
                v_hat_w = self.get_squared_grad_avg_weight_term_with_corrected_bias_for_layer(layer)
                # Squared gradient average term for biases with bias correction applied
                v_hat_b = self.get_squared_grad_avg_bias_term_with_corrected_bias_for_layer(layer)
                layer.update(self.get_change_of_weight_for_layer(m_hat_w, v_hat_w),
                             self.get_change_of_bias_for_layer(m_hat_b, v_hat_b))

    def update_momentum_weight_term_for_layer(self, layer: Layer) -> None:
        """
        Updates the momentum term for the weights of a specific layer.

        Implements the momentum update rule for the weights of a layer.
        It calculates the new momentum term based on previous term, the learning rate,
        the beta1 hyperparameter and the gradients of the weights.
        
        Parameters:
            layer (Layer): The layer for which the momentum weight term will be updated.
        """
        self.momentum_terms[layer]["w"] = self.beta1 * self.momentum_terms[layer]["w"]\
                                          + (1 - self.beta1) * layer.grad_weights

    def update_momentum_bias_term_for_layer(self, layer: Layer) -> None:
        """
        Update the momentum term for biases of a specific layer.

        Implements the momentum update rule for the biases of a layer.
        It calculates the new momentum term based on previous term,
        the beta1 hyperparameter and the gradients of the biases.
        
        Parameters:
            layer (Layer): The layer for which the momentum bias term will be updated.
        """
        self.momentum_terms[layer]["b"] = self.beta1 * self.momentum_terms[layer]["b"]\
                                          + (1 - self.beta1) * layer.grad_bias

    def update_squared_grad_avg_weight_term_for_layer(self, layer: Layer) -> None:
        """
        Update the squared gradient average term for weights for a specific layer.

        Implements the squared gradient average update rule for the weights of a layer.
        It calculates the new squared gradient average term based on the previous term,
        the beta2 hyperparameter and the gradients of the weights.

        Parameters:
            layer (Layer): The layer for which the average squared gradient 
                           weight term will be updated.
        """
        self.squared_grad_avg_terms[layer]["w"] =\
                    self.beta2 * self.squared_grad_avg_terms[layer]["w"]\
                    + (1 - self.beta2) * (layer.grad_weights * layer.grad_weights)

    def update_squared_grad_avg_bias_term_for_layer(self, layer: Layer) -> None:
        """
        Update the squared gradient average term for biases for a specific layer.

        Implements the squared gradient average update rule for the biases of a layer.
        It calculates the new squared gradient average term based on the previous term, 
        the beta2 hyperparameter and the gradients of the biases.

        Parameters:
            layer (Layer): The layer for which the average squared gradient 
                           bias term will be updated.        
        """
        self.squared_grad_avg_terms[layer]["b"] =\
                    self.beta2 * self.squared_grad_avg_terms[layer]["b"]\
                    + (1 - self.beta2) * (layer.grad_bias * layer.grad_bias)

    def get_momentum_weight_term_with_corrected_bias_for_layer(self, layer: Layer) -> np.ndarray:
        """
        Calculates and returns the momentum weight term for a layer,
        incorporating bias correction.

        Retrieves the momentum term for the weights of a specific layer
        and applies bias correction using the Adam algorithm formula.
        This correction helps to ensure numerical stability during training,
        especially in the early iterations.

        Parameters:
            layer (Layer): layer for which the term will be calculated.
        Returns:
            np.ndarray: The momentum weight term with corrected bias.
        """
        return self.momentum_terms[layer]["w"] / (1 - (self.beta1 ** self.time_step))

    def get_momentum_bias_term_with_corrected_bias_for_layer(self,
                                                             layer: Layer) -> np.ndarray:
        """
        Calculates and returns the momentum bias term for a layer,
        incorporating bias correction.

        Retrieves the momentum term for the biases of a specific layer
        and applies bias correction using the Adam algorithm formula.
        This correction helps to ensure numerical stability during training,
        especially in the early iterations.

        Parameters:
            layer (Layer): layer for which the term will be calculated.
        Return:
            np.ndarray: The momentum bias term with corrected bias.
        """
        return self.momentum_terms[layer]["b"] / (1 - (self.beta1 ** self.time_step))

    def get_squared_grad_avg_weight_term_with_corrected_bias_for_layer(self,
                                                                       layer: Layer) -> np.ndarray:
        """
        Calculates and returns the squared gradient average weight term for a layer,
        incorporating bias correction.

        Retrieves the squared gradient average term for the weights of a specific layer
        and applies bias correction using the Adam algorithm formula.
        This correction helps to ensure numerical stability during training,
        especially in the early iterations.

        Parameters:
            layer (Layer): layer for which the term will be calculated.
        Returns:
            np.ndarray: squared gradient average weight term with corrected bias.
        """
        return self.squared_grad_avg_terms[layer]["w"] / (1 - (self.beta2 ** self.time_step))

    def get_squared_grad_avg_bias_term_with_corrected_bias_for_layer(self,
                                                                     layer: Layer) -> np.ndarray:
        """
        Calculates and returns the squared gradient average bias term for a layer,
        incorporating bias correction.

        Retrieves the squared gradient average term for the biases of a specific layer
        and applies bias correction using the Adam algorithm formula.
        This correction helps to ensure numerical stability during training,
        especially in the early iterations.

        Parameters:
            layer (Layer): layer for which the term will be calculated.
        Returns:
            np.ndarray: squared gradient average bias term with corrected bias.
        """
        return self.squared_grad_avg_terms[layer]["b"] / (1 - (self.beta2 ** self.time_step))

    def get_change_of_weight_for_layer(self,
                                       m_hat_w: np.ndarray,
                                       v_hat_w: np.ndarray) -> np.ndarray:
        """
        Calculates and returns the change of weights for a layer.

        Uses the momentum term, squared gradient average term (after bias correction has 
        been applied to both) and the learning rate to calculate the change in weights
        for a specific layer. Adding the epsilon term will help to avoid zero division errors.
        
        Parameters:
            m_hat_w: momentum weight term with corrected bias for the layer
            v_hat_w: squared gradient average weight term with corrected bias for the layer
        Returns:
            np.ndarray: change of weights for the layer.
        """
        return (self.learning_rate * m_hat_w) / np.sqrt(v_hat_w + self.epsilon)

    def get_change_of_bias_for_layer(self,
                                     m_hat_b: np.ndarray,
                                     v_hat_b: np.ndarray) -> np.ndarray:
        """
        Calculates and returns the change of biases for a layer.

        Uses the momentum term and squared gradient average term for biases (after bias 
        correction has been applied to both) and the learning rate to calculate the 
        change in biases for a specific layer. Adding the epsilon term will help 
        to avoid zero division errors.

        Parameters:
            m_hat_w: momentum bias term with corrected bias for the layer
            v_hat_w: squared gradient average bias term with corrected bias for the layer
        Returns:
            np.ndarray: change of weights for the layer.
        """
        return (self.learning_rate * m_hat_b) / np.sqrt(v_hat_b + self.epsilon)

    # VALIDATION FUNCTIONS

    def validate_arguments(self,
                           learning_rate,
                           beta1,
                           beta2,
                           epsilon) -> None:
        """Validate parameters passed to Adam during initialization."""
        self.validate_learning_rate(learning_rate)
        self.validate_beta(beta1)
        self.validate_beta(beta2)
        self.validate_epsilon(epsilon)

    def validate_learning_rate(self, learning_rate: float) -> None:
        """
        Validates the learning rate parameter.

        Parameters:
            learning_rate (float)
        Raises:
            TypeError: If the learning rate is not of type float
            ValueError: If the learning rate is not a positive value
        """
        if not isinstance(learning_rate, float):
            raise TypeError(f"Learning rate has to be of type float. Got: {type(learning_rate)}")
        if learning_rate <= 0:
            raise ValueError(f"Learning rate has to be a positive value. Got {learning_rate}")

    def validate_beta(self, beta: float) -> None:
        """
        Validates beta1 and beta2 parameters.

        Parameters:
            beta (float): beta1/beta2
        Raises:
            TypeError: If beta is not of type float
            ValueError: If beta is not in range (0, 1)
        """
        if not isinstance(beta, float):
            raise TypeError("Hyperparameters beta1 and beta2 have to be of type float" +\
                            f"Got: {type(beta)}")
        if not 0 < beta < 1:
            raise ValueError("Hyperparameters beta1 and beta2 have to be in range (0, 1)" +\
                             f"Got: {beta}")

    def validate_epsilon(self, epsilon: float) -> None:
        """
        Validates epsilon parameter.

        Parameters:
            epsilon (float)
        Raises:
            TypeError: If epsilon is not of type float
            ValueError: If epsilon is not a positive value.
        """
        if not isinstance(epsilon, float):
            raise TypeError(f"Hyperparameter epsilon has to be of type float. Got {type(epsilon)}")
        if epsilon <= 0:
            raise ValueError(f"Hyperparameter epsilon has to be a positive value. Got: {epsilon}")
