import loss


class Model:
    def __init__(self, layers=[]) -> None:
        self.layers = layers
        self.loss = None
        self.loss_derivative = None

    def add_layer(self, layer):
        """Add a layer to the model."""
        self.layers.append(layer)

    def use_loss_function(self, function):
        """Define the loss function being used."""
        function = loss.get_loss(function)
        self.loss = function[0]
        self.loss_derivative = function[1]

    def predict(self, input_data):
        n_samples = len(input_data)
        result = []

        for i in range(n_samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)

        return result

    def fit(self, x_train, y_train, epochs, learning_rate, print_loss=True):
        """Train the model."""
        n_samples = len(x_train)

        for i in range(epochs):
            err = 0
            for j in range(n_samples):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward(output)
                err += self.loss(y_train[j], output)
                error = self.loss_derivative(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate)
            err /= n_samples
            if print_loss:
                print(f"Epoch: {i + 1}/{epochs}    error={err}")
