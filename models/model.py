from losses import loss
from exceptions import exception
from layers.input import Input
import numpy as np


class Model:
    def __init__(self, inputs, outputs) -> None:
        self.inputs = inputs
        self.outputs = outputs
        self.loss_fn = None

    def compile(self, loss_fn) -> None:
        # Add loss function
        self.loss_fn = loss.Loss(loss_fn)
        # Create a layer graph
        layers = [self.outputs]
        self.layers = []
        while layers:
            current = layers.pop(0)
            if not isinstance(current, Input):
                self.layers.append(current)
            if current.previous_layer and\
               current.previous_layer not in layers:
                layers.append(current.previous_layer)
        self.layers.reverse()
        self.validate()

    def predict(self, input_data):
        n_examples = len(input_data)
        result = np.zeros(n_examples)

        for i in range(n_examples):
            self.inputs.forward(input_data[i: i + 1])
            for layer in self.layers:
                layer.forward(layer.previous_layer.output)
            result[i] = self.outputs.output

        return result

    def fit(self, x_train,
            y_train, epochs,
            learning_rate,
            print_loss=True,
            batch_size=1):
        """Train the model."""
        n_batches = len(x_train) // batch_size

        for epoch in range(epochs):
            self.loss = 0
            for batch in range(n_batches):
                X_batch = x_train[batch * batch_size: (batch + 1) * batch_size]
                y_batch = y_train[batch * batch_size: (batch + 1) * batch_size]
                self.forward(X_batch)
                self.backward(y_batch)
                self.update(learning_rate)
            self.loss /= n_batches
            if print_loss:
                print(f"Epoch: {epoch + 1}/{epochs}\t" +
                      f"error={round(self.loss, 7)}")
        return self.loss

    def forward(self, x):
        self.inputs.forward(x)
        for layer in self.layers:
            layer.forward(layer.previous_layer.output)

    def backward(self, x):
        output = self.outputs.output
        self.loss += self.loss_fn.forward(x, output)
        derivative = self.loss_fn.backward(x, output)
        # Backward pass
        for layer in reversed(self.layers):
            derivative = layer.backward(derivative)

    def update(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)

    def validate(self):
        if self.loss_fn.name == "categorical_cross_entropy":
            if self.layers[-1].activation.name != "softmax":
                raise exception.IncompatibleLayerError(
                    "Categorical Cross Entropy loss should be preceded by " +
                    "Softmax activation function."
                )
        if self.layers[-1].activation and\
           self.layers[-1].activation.name == "softmax":
            if self.loss_fn.name != "categorical_cross_entropy":
                raise exception.IncompatibleLayerError(
                    "Softmax should be used together with Categorical " +
                    "Cross Entropy loss."
                )
        if "softmax" in [x.activation.name for x in self.layers[:-1]]:
            raise exception.IncompatibleLayerError(
                "Softmax should be used only as the activation function " +
                "for the last layer."
            )
        if not isinstance(self.inputs, Input):
            raise exception.IncompatibleLayerError(
                "Wrong layer type: use Input layer as the first layer of " +
                "your model."
            )
