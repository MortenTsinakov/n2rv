"""Model class."""

import numpy as np
from n2rv.exceptions import exception
from n2rv.layers.input import Input
from n2rv.layers.layer import Layer
from n2rv.losses import loss
from n2rv.metrics.metrics import Metrics
from n2rv.metrics.model_metrics import ModelMetrics
from n2rv.optimizers.optimizer import Optimizer


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
        self.metrics = None

    def compile(
        self, loss_fn: str, optimizer: Optimizer, metrics: list[Metrics] = None
    ) -> None:
        """
        Compile the model - define loss, optimizer and create layer graph.

        Inputs:
            loss_fn (str) - the name of the loss function that will be used
            optimizer (Optimizer) - the optimizer that will be used
            metrics (list[Metrics]) - list of Metrics class instances
        """
        self.loss_fn = loss.Loss(loss_fn)
        self.optimizer = optimizer
        self.metrics = ModelMetrics(metrics)
        # Create a layer graph
        layers = [self.outputs]
        while layers:
            current = layers.pop(0)
            if not isinstance(current, Input):
                self.layers.append(current)
            if current.previous_layer and current.previous_layer not in layers:
                layers.append(current.previous_layer)
        self.layers.reverse()
        self.validate_on_compile()

    def evaluate(self,
                 x_test: np.ndarray,
                 y_test: np.ndarray) -> dict[str, Metrics]:
        """
        Print out loss and metrics on test set.

        Inputs:
            x_test (np.ndarray) - test features
            y_test (np.ndarray) - test labels
        """
        pred = self.predict(x_test)
        loss = self.loss_fn.forward(y_test, pred)
        self.metrics.update_metrics(true=y_test, pred=pred)
        metric_dict = self.metrics.get_metrics(loss=loss)
        self.metrics.reset_metrics()
        return metric_dict

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict using the trained model.

        Inputs:
            x (np.ndarray) - the features based on which to predict
        Return:
            np.ndarray - numpy array of predictions
        """
        self.validate_on_predict(x)

        self.inputs.forward(input_data=x)
        for layer in self.layers:
            layer.forward(layer.previous_layer.output)

        return self.outputs.output

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int,
        print_metrics: bool = True,
        batch_size: int = 32,
    ) -> float:
        """
        Train the model.

        Inputs:
            x_train (np.ndarray) - features for training the model
            y_train (np.ndarray) - labels for calculating the loss on training
            epochs (int) - number of times that the training set will be
                iterated through
            print_metrics (bool) - print loss and metrics on every iteration
                if set True
            batch_size (int) - the number of samples to train on in one go
        Return:
            float - the training loss achieved by the end of training
        """
        self.validate_on_fit(x_train,
                             y_train, epochs,
                             print_metrics,
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
                self.metrics.update_metrics(y_batch, self.outputs.output)
            self.loss /= n_batches
            if print_metrics:
                self.print_metrics(epoch + 1, epochs)
            self.metrics.reset_metrics()
        return self.loss

    def forward(self, x: np.ndarray) -> None:
        """
        Forward pass through the network.

        Make some predictions to see how far off from the
        desired state the model is.

        Inputs:
            x (np.ndarray) - the input data
        """
        self.inputs.forward(x)
        for layer in self.layers:
            layer.forward(layer.previous_layer.output)

    def backward(self, y: np.ndarray) -> None:
        """
        Backward pass through the network.

        Calculates the gradients for each layer.

        Inputs:
            y (np.ndarray) - the true values
        """
        output = self.outputs.output
        self.loss += self.loss_fn.forward(y, output)
        derivative = self.loss_fn.backward(y, output)
        # Backward pass
        for layer in reversed(self.layers):
            derivative = layer.backward(derivative)

    def update(self) -> None:
        """Delegate updating the parameters to the optimizer."""
        self.optimizer.update(self.layers)

    def print_metrics(self, epoch: int, epochs: int) -> None:
        """
        Print metrics.

        Inputs:
            epoch (int) - current iteration
            epochs (int) - total iterations
        """
        metrics_dict = self.metrics.get_metrics(self.loss)
        metrics_to_print = []
        for k, v in metrics_dict.items():
            metrics_to_print.append(f"{k}: {v}")
        print(f"Epoch: {epoch}/{epochs} Train - {metrics_to_print}")

    # VALIDATION FUNCTIONS

    def validate_on_init(self, inputs: Layer, outputs: Layer) -> None:
        """
        Validate arguments provided on initialization.

        Inputs:
            inputs (Layer) - the first layer of the network
            outputs (Layer) - the last layer of the network
        """
        self.validate_correct_inputs_and_outputs(inputs, outputs)

    def validate_on_compile(self) -> None:
        """Validate arguments provided on compilation."""
        self.validate_optimizer()
        self.validate_softmax_and_cce_are_used_together()
        self.validate_path_between_inputs_and_outputs()

    def validate_on_predict(self, x: np.ndarray) -> None:
        """
        Validate arguments provided when predicting.

        Inputs:
            x (np.ndarray) - the features used for predicting
        """
        self.validate_input_data_type(x)
        self.validate_x_shape(x)

    def validate_on_fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int,
        print_loss: bool,
        batch_size: int,
    ) -> None:
        """
        Validate arguments provided on fitting.

        Inputs:
            x_train (np.ndarray) - features for training the model
            y_train (np.ndarray) - labels for calculating the loss on training
            epochs (int) - number of times that the training set will be
                iterated through
            print_metrics (bool) - print loss and metrics on every iteration
                if set True
            batch_size (int) - the number of samples to train on in one go
        """
        self.validate_input_data_type(x_train)
        self.validate_input_data_type(y_train)
        self.validate_x_shape(x_train)
        self.validate_y_shape(y_train)
        self.validate_x_y_same_length(x_train, y_train)
        self.validate_correct_epochs_arg(epochs)
        self.validate_correct_print_loss_arg(print_loss)
        self.validate_correct_batch_size_arg(batch_size, x_train)

    # VALIDATION HELPER FUNCTIONS

    def validate_correct_inputs_and_outputs(
        self, inputs: Layer, outputs: Layer
    ) -> None:
        """
        Make sure the inputs and outputs are of correct type.

        Inputs:
            inputs (Layer) - the first layer of the network
            outputs (Layer) - the last layer of the network
        """
        if inputs is None:
            raise ValueError("Couldn't initialize the Model: inputs can't be" +
                             " None")
        if outputs is None:
            raise ValueError(
                "Couldn't initialize the Model: outputs can't " + "be None."
            )
        if not isinstance(inputs, Input):
            raise exception.IncompatibleLayerError(
                "Wrong layer type: use Input layer as the first layer of "
                + "your model."
            )

    def validate_path_between_inputs_and_outputs(self) -> None:
        """
        Make sure that there's a connection between the first layer
        and the last layer of the network.
        """
        if self.layers[0].previous_layer != self.inputs:
            raise exception.DisconnectedLayersError(
                "No connection detected between inputs and " + "outputs"
            )

    def validate_softmax_and_cce_are_used_together(self) -> None:
        """Make sure that Softmax activation and Categorical Cross-Entropy
        loss are used together."""
        if self.loss_fn.name == "categorical_cross_entropy":
            if self.layers[-1].activation.name != "softmax":
                raise exception.IncompatibleLayerError(
                    "Categorical Cross Entropy loss should be preceded by "
                    + "Softmax activation function."
                )
        if self.layers[-1].activation and\
                self.layers[-1].activation.name == "softmax":
            if self.loss_fn.name != "categorical_cross_entropy":
                raise exception.IncompatibleLayerError(
                    "Softmax should be used together with Categorical "
                    + "Cross Entropy loss."
                )
        if "softmax" in [x.activation.name for x in self.layers[:-1]]:
            raise exception.IncompatibleLayerError(
                "Softmax should be used only as the activation function "
                + "for the last layer."
            )

    def validate_input_data_type(self, data: np.ndarray) -> None:
        """
        Make sure that input data is of the correct type.

        Inputs:
            data (np.ndarray) - the input data
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data should be a Numpy array." +
                            f"Got: {type(data)}")

    def validate_x_shape(self, x: np.ndarray) -> None:
        """
        Make sure that shape of the inputs matches the shape of the
        Input layer.

        Inputs:
            x (np.ndarray) - the input data
        """
        if x.shape[1:] != self.inputs.shape:
            raise exception.ShapeMismatchError(
                "Mismatch between shapes of data fed into the model and "
                + f"Input layer detected: {x.shape[1:]} != {self.inputs.shape}"
            )

    def validate_y_shape(self, y: np.ndarray) -> None:
        """
        Make sure that the labels shape is matching the output shape.

        Inputs:
            y (np.ndarray) - the labels
        """
        if y.shape[-1] != self.outputs.output_size:
            raise exception.ShapeMismatchError(
                "Mismatch between shapes of labels fed into the model and "
                + f"output layer detected: {y.shape[-1]} != "
                + f"{self.outputs.output_size}"
            )

    def validate_x_y_same_length(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Make sure that there are same amount of samples in training
        data and labels.

        Inputs:
            x (np.ndarray) - training features
            y (np.ndarray) - training labels
        """
        if len(x) != len(y):
            raise exception.ShapeMismatchError(
                "X_train and y_train should be the same length. "
                + f"Got: len(x_train): {len(x)}, len(y_train): {len(y)}"
            )

    def validate_correct_epochs_arg(self, epochs: int) -> None:
        """
        Make sure that epochs is of type int.

        Inputs:
            epochs (int) - number of times that the training set will be
                iterated through
        """
        if not isinstance(epochs, int):
            raise TypeError(
                "Epochs argument in Model.fit() should be an integer. "
                + f"Got {type(epochs)}"
            )
        if epochs <= 0:
            raise ValueError(
                "Epochs argument in Model.fit() should be > 0. " +
                f"Got {epochs}"
            )

    def validate_correct_print_loss_arg(self, print_metrics: bool) -> None:
        """
        Make sure print_loss argument is of type boolean.

        Inputs:
            print_metrics (bool) - variable to determine whether metrics will
                be printed on each epoch
        """
        if not isinstance(print_metrics, bool):
            raise TypeError(
                "Print loss argument in Model.fit() should be "
                + f"of boolean type. Got: {type(print_metrics)}"
            )

    def validate_correct_batch_size_arg(self,
                                        batch_size: int,
                                        x: np.ndarray) -> None:
        """
        Make sure batch size argument is positive integer less than the
        size of training data.

        Inputs:
            batch_size (int) - the number of samples on which to train on
                in one go
            x (np.ndarray) - the training features
        """
        if not isinstance(batch_size, int):
            raise TypeError(
                "Batch size argument in Model.fit() should be an "
                + f"integer. Got {type(batch_size)}"
            )
        if batch_size <= 0:
            raise ValueError(
                "Batch size argment in Model.fit() should be >0." +
                f"Got {batch_size}"
            )
        if batch_size > len(x):
            raise ValueError(
                "Batch size argument in Model.fit() should be "
                + f"<len(x_train). Got: {batch_size}"
            )

    def validate_optimizer(self) -> None:
        """
        Make sure that the optimizer argument provided
        is of type Optimizer.
        """
        if not isinstance(self.optimizer, Optimizer):
            raise TypeError(
                "Optimizer has to be of type Optimizer."
                + f"Got: {type(self.optimizer)}"
            )
