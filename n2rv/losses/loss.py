"""Loss class provided to the model."""

import numpy as np
from n2rv.losses import loss_functions


class Loss:
    """Loss class."""

    def __init__(self, loss_function: str) -> None:
        loss = loss_functions.get_loss(loss_function)
        self.name = loss_function
        self.loss = loss[0]
        self.loss_derivative = loss[1]

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the loss.

        Inputs:
            y_true (np.ndarray) - the true values
            y_pred (np.ndarray) - the predicted values
        Return:
            float - the average error between the actual values and
                the predicted values.
        """
        return self.loss(y_true, y_pred)

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate the derivative of the loss function.

        Inputs:
            y_true (np.ndarray) - the true values
            y_pred (np.ndarray) - the predicted values
        Return:
            np.ndarray - the derivative of the loss function evaluated
                at y_pred.
        """
        return self.loss_derivative(y_true, y_pred)
