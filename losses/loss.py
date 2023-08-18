from losses import loss_functions


class Loss:
    def __init__(self, loss_function) -> None:
        loss = loss_functions.get_loss(loss_function)
        self.loss = loss[0]
        self.loss_derivative = loss[1]

    def forward(self, y_true, y_pred):
        return self.loss(y_true, y_pred)

    def backward(self, y_true, y_pred):
        return self.loss_derivative(y_true, y_pred)
