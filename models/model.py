from losses import loss
from exceptions import exception
from layers.input import Input
import numpy as np


class Model:
    def __init__(self, inputs, outputs) -> None:
        self.validate_on_init(inputs, outputs)
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
        self.validate_on_compile()

    def predict(self, input_data):
        self.validate_on_predict(input_data)
        n_examples = len(input_data)
        result = np.zeros(n_examples)

        for i in range(n_examples):
            self.inputs.forward(input_data[i: i + 1])
            for layer in self.layers:
                layer.forward(layer.previous_layer.output)
            result[i] = self.outputs.output

        return result

    def fit(self,
            x_train,
            y_train,
            epochs,
            learning_rate,
            print_loss=True,
            batch_size=1):
        """Train the model."""
        self.validate_on_fit(x_train,
                             y_train,
                             epochs,
                             learning_rate,
                             print_loss,
                             batch_size)
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

    # VALIDATION FUNCTIONS

    def validate_on_init(self, inputs, outputs):
        self.validate_correct_inputs_and_outputs(inputs, outputs)

    def validate_on_compile(self):
        self.validate_softmax_and_cce_are_used_together()
        self.validate_path_between_inputs_and_outputs()

    def validate_on_predict(self, X):
        self.validate_input_data_type(X)
        self.validate_X_shape(X)

    def validate_on_fit(self,
                        x_train,
                        y_train,
                        epochs,
                        learning_rate,
                        print_loss,
                        batch_size):
        self.validate_input_data_type(x_train)
        self.validate_input_data_type(y_train)
        self.validate_X_shape(x_train)
        self.validate_y_shape(y_train)
        self.validate_X_y_same_length(x_train, y_train)
        self.validate_correct_epochs_arg(epochs)
        self.validata_correct_lr_arg(learning_rate)
        self.validate_correct_print_loss_arg(print_loss)
        self.validate_correct_batch_size_arg(batch_size, x_train)

    # VALIDATION HELPER FUNCTIONS

    def validate_correct_inputs_and_outputs(self, inputs, outputs):
        if inputs is None:
            raise ValueError("Couldn't initialize the Model: inputs can't be" +
                             " None")
        if outputs is None:
            raise ValueError("Couldn't initialize the Model: outputs can't " +
                             "be None.")
        if not isinstance(inputs, Input):
            raise exception.IncompatibleLayerError(
                "Wrong layer type: use Input layer as the first layer of " +
                "your model."
            )

    def validate_path_between_inputs_and_outputs(self):
        if self.layers[0].previous_layer != self.inputs:
            raise exception.DisconnectedLayersError(
                "No connection detected between inputs and " +
                "outputs"
            )

    def validate_softmax_and_cce_are_used_together(self):
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

    def validate_input_data_type(self, data):
        if type(data) != np.ndarray:
            raise TypeError(
                f"Input data should be a Numpy array. Got: {type(data)}"
            )

    def validate_X_shape(self, X):
        if X.shape[1:] != self.inputs.shape:
            raise exception.ShapeMismatchError(
                "Mismatch between shapes of data fed into the model and " +
                f"Input layer detected: {X.shape[1:]} != {self.inputs.shape}"
            )

    def validate_y_shape(self, y):
        if y.shape[-1] != self.outputs.output_size:
            raise exception.ShapeMismatchError(
                "Mismatch between shapes of labels fed into the model and " +
                f"output layer detected: {y.shape[-1]} != " +
                f"{self.outputs.output_size}"
            )

    def validate_X_y_same_length(self, X, y):
        if len(X) != len(y):
            raise exception.ShapeMismatchError(
                "X_train and y_train should be the same length. " +
                f"Got: len(x_train): {len(X)}, len(y_train): {len(y)}"
            )

    def validate_correct_epochs_arg(self, epochs):
        if type(epochs) != int:
            raise TypeError(
                "Epochs argument in Model.fit() should be an integer. " +
                f"Got {type(epochs)}"
            )
        if epochs <= 0:
            raise ValueError(
                "Epochs argument in Model.fit() should be > 0. " +
                f"Got {epochs}"
            )

    def validata_correct_lr_arg(self, learning_rate):
        if type(learning_rate) != float:
            raise TypeError(
                "Learning rate argument in Model.fit() " +
                f"should be a float. Got: {type(learning_rate)}")
        if learning_rate <= 0:
            raise ValueError(
                "Learning rate argument in Model.fit() " +
                f"should be >0. Got: {learning_rate}"
            )

    def validate_correct_print_loss_arg(self, print_loss):
        if type(print_loss) != bool:
            raise TypeError(
                "Print loss argument in Model.fit() should be " +\
                f"of boolean type. Got: {type(print_loss)}"
            )

    def validate_correct_batch_size_arg(self, batch_size, X):
        if type(batch_size) != int:
            raise TypeError(
                "Batch size argument in Model.fit() should be an " +
                f"integer. Got {type(batch_size)}"
            )
        if batch_size <= 0:
            raise ValueError(
                "Batch size argment in Model.fit() should be >0." +
                f"Got {batch_size}"
            )
        if batch_size > len(X):
            raise ValueError(
                "Batch size argument in Model.fit() should be " +
                f"<len(x_train). Got: {batch_size}"
            )
