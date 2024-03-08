"""Tests for activations functions."""


import os
import sys
import unittest

import numpy as np

# Get the absolute path of the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Paths to files in the directories above
paths = [
    os.path.join(current_script_dir, '..', 'activations')
]

# Add all paths
sys.path += [os.path.dirname(name) for name in paths]

# Import files for testing
from n2rv.activations import activation_functions as af

np.random.seed(0)


class TestActivations(unittest.TestCase):
    """Tests for activation functions."""

    def test_tanh(self):
        """
        Test - compare the outputs of Tanh activation function to
        manually calculated values.
        """
        x = np.array([
            [0.6, -0.2, 0.0],
            [1.5, -2.0, -0.8],
            [0.6, -1.2, 1.5]
        ])

        desired = np.array([
            [0.537049567, -0.1973753202, 0.0],
            [0.9051482536, -0.9640275801, -0.6640367703],
            [0.5370495670, -0.8336546070, 0.9051482536]
        ])

        actual = af.tanh(x)

        np.testing.assert_allclose(actual=actual, desired=desired)

    def test_tanh_derivative(self):
        """
        Test - compare the outputs of Tanh activation derivative function to
        manually calculated values.
        """
        x = np.array([
            [0.6, -0.2, 0.0],
            [1.5, -2.0, -0.8],
            [0.6, -1.2, 1.5]
        ])
        output_error = np.array([
            [0.1, 0.2, 0.3],
            [-0.1, -0.2, -0.3],
            [1.2, -1.5, 2.5]
        ])
        desired = np.array([
            [0.07115777626, 0.1922085966, 0.3],
            [-0.01807066389, -0.01413016497, -0.1677165503],
            [0.8538933151, -0.4575299943, 0.4517665973]
        ])

        actual = af.tanh_derivative(x=x, output_error=output_error)

        np.testing.assert_allclose(actual=actual, desired=desired)

    def test_relu(self):
        """
        Test - compare the outputs of ReLU activation function to
        manually calculated values.
        """
        x = np.array([
            [0.6, -0.2, 0.0],
            [1.5, -2.0, -0.8],
            [0.6, -1.2, 1.5]
        ])

        desired = np.array([
            [0.6, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [0.6, 0.0, 1.5]
        ])

        actual = af.relu(x=x)

        np.testing.assert_allclose(actual=actual, desired=desired)

    def test_relu_derivative(self):
        """
        Test - compare the outputs of ReLU activation derivative function to
        manually calculated values.
        """
        x = np.array([
            [0.6, -0.2, 0.0],
            [1.5, -2.0, -0.8],
            [0.6, -1.2, 1.5]
        ])

        output_error = np.array([
            [0.1, 0.2, 0.3],
            [-0.1, -0.2, -0.3],
            [1.2, -1.5, 2.5]
        ])

        desired = np.array([
            [0.1, 0.0, 0.3],
            [-0.1, 0.0, 0.0],
            [1.2, 0.0, 2.5]
        ])

        actual = af.relu_derivative(x=x, output_error=output_error)

        np.testing.assert_allclose(actual=actual, desired=desired)

    def test_leaky_relu(self):
        """
        Test - compare the outputs of Leaky ReLU activation function to
        manually calculated values.
        """
        x = np.array([
            [0.6, -0.2, 0.0],
            [1.5, -2.0, -0.8],
            [0.6, -1.2, 1.5]
        ])

        desired = np.array([
            [0.6, -0.002, 0.0],
            [1.5, -0.02, -0.008],
            [0.6, -0.012, 1.5]
        ])

        actual = af.leaky_relu(x=x)

        np.testing.assert_allclose(actual=actual, desired=desired)

    def test_leaky_relu_derivative(self):
        """
        Test - compare the outputs of Leaky ReLU activation derivative
        function to manually calculated values.
        """
        x = np.array([
            [0.6, -0.2, 0.0],
            [1.5, -2.0, -0.8],
            [0.6, -1.2, 1.5]
        ])

        output_error = np.array([
            [0.1, 0.2, 0.3],
            [-0.1, -0.2, -0.3],
            [1.2, -1.5, 2.5]
        ])

        desired = np.array([
            [0.1, 0.002, 0.3],
            [-0.1, -0.002, -0.003],
            [1.2, -0.015, 2.5]
        ])

        acutal = af.leaky_relu_derivative(x=x, output_error=output_error)

        np.testing.assert_allclose(actual=acutal, desired=desired)

    def test_sigmoid(self):
        """
        Test - compare the outputs of Sigmoid activation function to
        manually calculated values.
        """
        x = np.array([
            [0.6, -0.2, 0.0],
            [1.5, -2.0, -0.8],
            [0.6, -1.2, 1.5]
        ])

        desired = np.array([
            [0.6456563062, 0.4501660027, 0.5],
            [0.8175744762, 0.1192029220, 0.3100255189],
            [0.6456563062, 0.2314752165, 0.8175744762]
        ])

        actual = af.sigmoid(x=x)

        np.testing.assert_allclose(actual=actual, desired=desired)

    def test_sigmoid_derivative(self):
        """
        Test - compare the outputs of Sigmoid activation derivative
        function to manually calculated values.
        """
        x = np.array([
            [0.6, -0.2, 0.0],
            [1.5, -2.0, -0.8],
            [0.6, -1.2, 1.5]
        ])

        output_error = np.array([
            [0.1, 0.2, 0.3],
            [-0.1, -0.2, -0.3],
            [1.2, -1.5, 2.5]
        ])

        desired = np.array([
            [0.02287842405, 0.04950331454, 0.075],
            [-0.01491464521, -0.02099871708, -0.06417290896],
            [0.2745410885, -0.2668416610, 0.3728661302]
        ])

        actual = af.sigmoid_derivative(x=x, output_error=output_error)

        np.testing.assert_allclose(actual=actual, desired=desired)

    def test_linear(self):
        """
        Test - compare the outputs of Linear activation function to
        manually calculated values.
        """
        x = np.array([
            [0.6, -0.2, 0.0],
            [1.5, -2.0, -0.8],
            [0.6, -1.2, 1.5]
        ])

        actual = af.linear(x=x)

        np.testing.assert_allclose(actual=actual, desired=x)

    def test_linear_derivative(self):
        """
        Test - compare the outputs of Linear activation derivative
        function to manually calculated values.
        """
        x = np.array([
            [0.6, -0.2, 0.0],
            [1.5, -2.0, -0.8],
            [0.6, -1.2, 1.5]
        ])

        output_error = np.array([
            [0.1, 0.2, 0.3],
            [-0.1, -0.2, -0.3],
            [1.2, -1.5, 2.5]
        ])

        actual = af.linear_derivative(x=x, output_error=output_error)

        np.testing.assert_allclose(actual=actual, desired=output_error)

    def test_softmax_cce(self):
        """
        Test - compare the outputs of Softmax activation function to
        manually calculated values.
        """
        x = np.array([
            [0.6, -0.2, 0.0],
            [1.5, -2.0, -0.8],
            [0.6, -1.2, 1.5]
        ])

        desired = np.array([
            [0.5004652825, 0.2248735470, 0.2746611705],
            [0.8845986036, 0.02671256321, 0.08868883316],
            [0.2758695270, 0.04560092611, 0.6785295469]
        ])

        actual = af.softmax_with_categorical_cross_entropy(x=x)

        np.testing.assert_allclose(actual=actual, desired=desired)

    def test_softmax_cce_derivative(self):
        """
        Test - compare the outputs of Softmax activation function to
        manually calculated values.
        """
        x = np.array([
            [0.6, -0.2, 0.0],
            [1.5, -2.0, -0.8],
            [0.6, -1.2, 1.5]
        ])

        output_error = np.array([
            [0.1, 0.2, 0.3],
            [-0.1, -0.2, -0.3],
            [1.2, -1.5, 2.5]
        ])

        # Because the gradients are calculated by the CCE derivative function
        # we expect back only the output error (no calculation in this one)

        actual = af.softmax_with_categorical_cross_entropy_derivative(
            x=x,
            output_error=output_error
        )

        np.testing.assert_allclose(actual=actual, desired=output_error)

    def test_get_function_correct_function_name(self):
        """Test - Valid activation names should not throw an exception."""
        try:
            af.get_function('tanh')
        except ValueError:
            self.fail("activation_functions.get_function() should not throw" +
                      " a ValueError with argument: tanh")
        try:
            af.get_function('relu')
        except ValueError:
            self.fail("activation_functions.get_function() should not throw" +
                      " a ValueError with argument: relu")
        try:
            af.get_function('sigmoid')
        except ValueError:
            self.fail("activation_functions.get_function() should not throw" +
                      " a ValueError with argument: sigmoid")
        try:
            af.get_function('softmax')
        except ValueError:
            self.fail("activation_functions.get_function() should not throw" +
                      " a ValueError with argument: softmax")

    def test_get_function_incorrect_function_name(self):
        """Test - Invalid activation names should thrown an exception."""
        try:
            af.get_function('mse')
            self.fail("activation_function.get_function() should not allow" +
                      " argument: 'mse'")
        except ValueError:
            pass
        try:
            af.get_function('')
            self.fail("activation_function.get_function() should not allow" +
                      " empty string argument.")
        except ValueError:
            pass
        try:
            af.get_function(None)
            self.fail("activation_function.get_function() should not allow" +
                      " None argument")
        except ValueError:
            pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
