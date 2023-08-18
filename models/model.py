import losses.loss as loss


class Model:
    def __init__(self, inputs, outputs) -> None:
        self.inputs = inputs
        self.outputs = outputs
        self.loss = None
        self.loss_derivative = None

    def compile(self, loss_fn) -> None:
        # Add loss function
        loss_fn = loss.get_loss(loss_fn)
        self.loss = loss_fn[0]
        self.loss_derivative = loss_fn[1]
        # Create a layer graph
        outputs = [x for x in self.outputs]
        self.layers = []
        while outputs:
            current = outputs.pop(0)
            self.layers.append(current)
            if current.previous_layer and\
               current.previous_layer not in outputs:
                outputs.append(current.previous_layer)

    def predict(self, input_data):
        n_samples = len(input_data)
        result = []

        for i in range(n_samples):
            self.inputs.forward(input_data[i])
            for layer in reversed(self.layers[:-1]):
                layer.forward(layer.previous_layer.output)
            result.append(self.layers[0].output)

        return result

    def fit(self, x_train, y_train, epochs, learning_rate, print_loss=True):
        """Train the model."""
        n_samples = len(x_train)

        for i in range(epochs):
            self.err = 0
            for j in range(n_samples):
                self.forward(x_train[j])
                self.backward(y_train[j])
                self.update(learning_rate)
            self.err /= n_samples
            if print_loss:
                print(f"Epoch: {i + 1}/{epochs}    error={round(self.err, 7)}")
        return self.err

    def forward(self, x):
        self.inputs.forward(x)
        for layer in reversed(self.layers[:-1]):
            layer.forward(layer.previous_layer.output)

    def backward(self, x):
        # Calculate loss
        output = self.layers[0].output
        self.err += self.loss(x, output)
        error = self.loss_derivative(x, output)
        # Backward pass
        for layer in self.layers:
            error = layer.backward(error)

    def update(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)
