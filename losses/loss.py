"""Loss class provided to the model."""


from losses import loss_functions


class Loss:
    """Loss class."""
    def __init__(self, loss_function: str) -> None:
        loss = loss_functions.get_loss(loss_function)
        self.name = loss_function
        self.loss = loss[0]
        self.loss_derivative = loss[1]

    def forward(self, y_true, y_pred):
        """Calculate the loss."""
        return self.loss(y_true, y_pred)

    def backward(self, y_true, y_pred):
        """Calculate the derivative of the loss function."""
        return self.loss_derivative(y_true, y_pred)
