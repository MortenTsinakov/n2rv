"""Tests for activations functions."""


import unittest
import os
import sys
import numpy as np
# Don't print Tensorflow logs about GPUs and other such things.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# Get the absolute path of the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Paths to files in the directories above
paths = [
    os.path.join(current_script_dir, '..', 'activations')
]

# Add all paths
sys.path += [os.path.dirname(name) for name in paths]

# Import files for testing
from n2rv.activations import activation_functions

np.random.seed(0)


class TestActivations(unittest.TestCase):
    """Tests for activation functions."""
    def test_tanh(self):
        """Test - Compare Tanh function to Tensorflow Tanh function."""
        x = np.random.randn(10)

        tf_tanh = tf.tanh(x).numpy()
        custom_tanh = activation_functions.tanh(x)

        np.testing.assert_allclose(custom_tanh, tf_tanh, 1e-5)

    def test_relu(self):
        """Test - Comapare ReLU function to Tensorflow ReLU function"""
        x = np.random.randn(10)

        tf_relu = tf.nn.relu(x).numpy()
        custom_relu = activation_functions.relu(x)

        np.testing.assert_allclose(custom_relu, tf_relu, 1e-5)

    def test_sigmoid(self):
        """Test - Compare Sigmoid function to Tensorflow Sigmoid function."""
        x = np.random.randn(10)

        tf_sigmoid = tf.sigmoid(x).numpy()
        custom_sigmoid = activation_functions.sigmoid(x)

        np.testing.assert_allclose(custom_sigmoid, tf_sigmoid, 1e-5)

    def test_tanh_gradients(self):
        """Test - Compare Tanh derivatives to Tensorflow Tanh derivatives."""
        x = np.random.randn(10)

        tf_x = tf.constant(x, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(tf_x)
            tf_tanh = tf.tanh(tf_x)

        tf_grads = tape.gradient(tf_tanh, tf_x).numpy()
        custom_grads = activation_functions.tanh_derivative(x, 1)

        np.testing.assert_allclose(custom_grads, tf_grads, 1e-5)

    def test_relu_gradients(self):
        """Test - Compare ReLU derivatives to Tensorflow ReLU derivatives."""
        x = np.random.randn(10)

        tf_x = tf.constant(x, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(tf_x)
            tf_relu = tf.nn.relu(tf_x)

        tf_grads = tape.gradient(tf_relu, tf_x).numpy()
        custom_grads = activation_functions.relu_derivative(x, 1)

        np.testing.assert_allclose(custom_grads, tf_grads, 1e-5)

    def test_sigmoid_gradients(self):
        """Test - Compare Sigmoid derivatives to Tensorflow Sigmoid derivatives."""
        x = np.random.randn(10)

        tf_x = tf.constant(x, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(tf_x)
            tf_sigmoid = tf.sigmoid(tf_x)

        tf_grads = tape.gradient(tf_sigmoid, tf_x).numpy()
        custom_grads = activation_functions.sigmoid_derivative(x, 1)
        np.testing.assert_allclose(custom_grads, tf_grads, 1e-5)

    def test_get_function_correct_function_name(self):
        """Test - Valid activation names should not throw an exception."""
        try:
            activation_functions.get_function('tanh')
        except ValueError:
            self.fail("activation_functions.get_function() should not throw" +
                      " a ValueError with argument: tanh")
        try:
            activation_functions.get_function('relu')
        except ValueError:
            self.fail("activation_functions.get_function() should not throw" +
                      " a ValueError with argument: relu")
        try:
            activation_functions.get_function('sigmoid')
        except ValueError:
            self.fail("activation_functions.get_function() should not throw" +
                      " a ValueError with argument: sigmoid")
        try:
            activation_functions.get_function('softmax')
        except ValueError:
            self.fail("activation_functions.get_function() should not throw" +
                      " a ValueError with argument: softmax")

    def test_get_function_incorrect_function_name(self):
        """Test - Invalid activation names should thrown an exception."""
        try:
            activation_functions.get_function('mse')
            self.fail("activation_function.get_function() should not allow" +
                      " argument: 'mse'")
        except ValueError:
            pass
        try:
            activation_functions.get_function('')
            self.fail("activation_function.get_function() should not allow" +
                      " empty string argument.")
        except ValueError:
            pass
        try:
            activation_functions.get_function(None)
            self.fail("activation_function.get_function() should not allow" +
                      " None argument")
        except ValueError:
            pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
