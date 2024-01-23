"""Model class."""


import numpy as np
from losses import loss
from exceptions import exception
from layers.input import Input
from layers.layer import Layer
from optimizers.optimizer import Optimizer


class Model:
    """Model class."""
    def __init__(self, inputs: Layer, outputs: Layer) -> None:
        self.validate_on_init(inputs, outputs)
        self.inputs = inputs
        self.outputs = outputs
        self.loss_fn = None
        self.optimizer = None
        self.layers = []
        self.loss = 0

    def compile(self, loss_fn: str, optimizer: Optimizer) -> None:
        """Compile the model - define loss, optimizer and create layer graph."""
        self.loss_fn = loss.Loss(loss_fn)
        self.optimizer = optimizer
        # Create a layer graph
        layers = [self.outputs]
        while layers:
            current = layers.pop(0)
            if not isinstance(current, Input):
                self.layers.append(current)
            if current.previous_layer and\
               current.previous_layer not in layers:
                layers.append(current.previous_layer)
        self.layers.reverse()
        self.validate_on_compile()

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Predict using the trained model."""
        self.validate_on_predict(input_data)
        n_examples = len(input_data)
        result = []

        for i in range(n_examples):
            self.inputs.forward(input_data[i: i + 1])
            for layer in self.layers:
                layer.forward(layer.previous_layer.output)
            result.append(self.outputs.output)

        return result

    def fit(self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            epochs: int,
            print_loss: bool =True,
            batch_size: int =1) -> np.ndarray:
        """Train the model."""
        self.validate_on_fit(x_train,
                             y_train,
                             epochs,
                             print_loss,
                             batch_size)
        n_batches = len(x_train) // batch_size
        if len(x_train) % batch_size != 0:
            n_batches += 1

        for epoch in range(epochs):
            self.loss = 0
            for batch in range(n_batches):
                x_batch = x_train[batch * batch_size: (batch + 1) * batch_size]
                y_batch = y_train[batch * batch_size: (batch + 1) * batch_size]
                self.forward(x_batch)
                self.backward(y_batch)
                self.update()
            self.loss /= n_batches
            if print_loss:
                print(f"Epoch: {epoch + 1}/{epochs}\t" +
                      f"error={round(self.loss, 7)}")
        return self.loss

    def forward(self, x: np.ndarray) -> None:
        """Forward pass through the network."""
        self.inputs.forward(x)
        for layer in self.layers:
            layer.forward(layer.previous_layer.output)

    def backward(self, y: np.ndarray) -> None:
        """Backward pass through the network."""
        output = self.outputs.output
        self.loss += self.loss_fn.forward(y, output)
        derivative = self.loss_fn.backward(y, output)
        # Backward pass
        for layer in reversed(self.layers):
            derivative = layer.backward(derivative)

    def update(self) -> None:
        """Delegate updating the parameters to the optimizer."""
        self.optimizer.update(self.layers)

    # VALIDATION FUNCTIONS

    def validate_on_init(self, inputs: Layer, outputs: Layer) -> None:
        """Validate arguments provided on initialization."""
        self.validate_correct_inputs_and_outputs(inputs, outputs)

    def validate_on_compile(self) -> None:
        """Validate arguments provided on compilation."""
        self.validate_optimizer()
        self.validate_softmax_and_cce_are_used_together()
        self.validate_path_between_inputs_and_outputs()

    def validate_on_predict(self, x: np.ndarray) -> None:
        """Validate arguments provided when predicting."""
        self.validate_input_data_type(x)
        self.validate_x_shape(x)

    def validate_on_fit(self,
                        x_train: np.ndarray,
                        y_train: np.ndarray,
                        epochs: int,
                        print_loss: bool,
                        batch_size: int) -> None:
        """Validate arguments provided on fitting."""
        self.validate_input_data_type(x_train)
        self.validate_input_data_type(y_train)
        self.validate_x_shape(x_train)
        self.validate_y_shape(y_train)
        self.validate_x_y_same_length(x_train, y_train)
        self.validate_correct_epochs_arg(epochs)
        self.validate_correct_print_loss_arg(print_loss)
        self.validate_correct_batch_size_arg(batch_size, x_train)

    # VALIDATION HELPER FUNCTIONS

    def validate_correct_inputs_and_outputs(self, inputs: Layer, outputs: Layer) -> None:
        """Make sure the inputs and outputs are of correct type."""
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

    def validate_path_between_inputs_and_outputs(self) -> None:
        """Make sure that the input layer and output layer are connected through other layers in the network."""
        if self.layers[0].previous_layer != self.inputs:
            raise exception.DisconnectedLayersError(
                "No connection detected between inputs and " +
                "outputs"
            )

    def validate_softmax_and_cce_are_used_together(self) -> None:
        """Make sure that Softmax activation and Categorical Cross-Entropy loss are used together."""
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

    def validate_input_data_type(self, data: np.ndarray) -> None:
        """Make sure that input data is of the correct type."""
        if not isinstance(data, np.ndarray):
            raise TypeError(
                f"Input data should be a Numpy array. Got: {type(data)}"
            )

    def validate_x_shape(self, x: np.ndarray) -> None:
        """Make sure that shape of the inputs matches the shape of the Input layer."""
        if x.shape[1:] != self.inputs.shape:
            raise exception.ShapeMismatchError(
                "Mismatch between shapes of data fed into the model and " +
                f"Input layer detected: {x.shape[1:]} != {self.inputs.shape}"
            )

    def validate_y_shape(self, y: np.ndarray) -> None:
        """Make sure that the labels shape is matching the output shape."""
        if y.shape[-1] != self.outputs.output_size:
            raise exception.ShapeMismatchError(
                "Mismatch between shapes of labels fed into the model and " +
                f"output layer detected: {y.shape[-1]} != " +
                f"{self.outputs.output_size}"
            )

    def validate_x_y_same_length(self, x: np.ndarray, y: np.ndarray) -> None:
        """Make sure that there are same amount of samples in training data and labels."""
        if len(x) != len(y):
            raise exception.ShapeMismatchError(
                "X_train and y_train should be the same length. " +
                f"Got: len(x_train): {len(x)}, len(y_train): {len(y)}"
            )

    def validate_correct_epochs_arg(self, epochs: int) -> None:
        """Make sure that epochs is of type int."""
        if not isinstance(epochs, int):
            raise TypeError(
                "Epochs argument in Model.fit() should be an integer. " +
                f"Got {type(epochs)}"
            )
        if epochs <= 0:
            raise ValueError(
                "Epochs argument in Model.fit() should be > 0. " +
                f"Got {epochs}"
            )

    def validate_correct_print_loss_arg(self, print_loss: bool) -> None:
        """Make sure print_loss argument is of type boolean."""
        if not isinstance(print_loss, bool):
            raise TypeError(
                "Print loss argument in Model.fit() should be " +\
                f"of boolean type. Got: {type(print_loss)}"
            )

    def validate_correct_batch_size_arg(self, batch_size: int, x: np.ndarray) -> None:
        """Make sure batch size argument is positive integer less than the size of training data."""
        if not isinstance(batch_size, int):
            raise TypeError(
                "Batch size argument in Model.fit() should be an " +
                f"integer. Got {type(batch_size)}"
            )
        if batch_size <= 0:
            raise ValueError(
                "Batch size argment in Model.fit() should be >0." +
                f"Got {batch_size}"
            )
        if batch_size > len(x):
            raise ValueError(
                "Batch size argument in Model.fit() should be " +
                f"<len(x_train). Got: {batch_size}"
            )
        
    def validate_optimizer(self) -> None:
        """Make sure that the optimizer argument provided is of type Optimizer."""
        if not isinstance(self.optimizer, Optimizer):
            raise TypeError(
                f"Optimizer has to be of type Optimizer. Got: {type(self.optimizer)}"
            )
