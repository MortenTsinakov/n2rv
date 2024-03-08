"""Base layer for all the different layers available."""

from numpy import ndarray


class Layer:
    """Base layer class for other layers."""

    def forward(self, input_data: ndarray) -> ndarray:
        """
        Calculate values on forward pass through the layer.

        Inputs:
            input_data (np.ndarray) - the data coming into the layer
        Return:
            np.ndarray - data after necessary calculations have been performed
                         on the input data
        """
        raise NotImplementedError

    def backward(self, output_error: ndarray) -> ndarray:
        """
        Calculate gradients for weights and biases and error for the
        previous layer in the network.

        Inputs:
            output_error (np.ndarray) - error in the output
        Return:
            np.ndarray - error in the inputs
        """
        raise NotImplementedError

    def update(self, weights_change: ndarray, bias_change: ndarray) -> None:
        """
        Update weights and biases according to the values calculated by
        the model optimizer.

        Inputs:
            weights_change (np.ndarray) - the amounts that should be
                subtracted from the layer weights
            bias_chagne (np.ndarray) - the amounts that should be subtracted
                from the layer biases
        """
        raise NotImplementedError
