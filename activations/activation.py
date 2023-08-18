from layers.layer import Layer
import activations.activation_functions as activation_functions


class Activation(Layer):
    def __init__(self, activation_function) -> None:
        activation = activation_functions.get_function(activation_function)
        self.activation = activation[0]
        self.activation_derivative = activation[1]

    def forward(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward(self, output_error):
        return self.activation_derivative(self.input) * output_error
