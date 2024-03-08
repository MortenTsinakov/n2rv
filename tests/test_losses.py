"""Tests for loss functions."""


import os
import sys
import unittest

import numpy as np

# Get the absolute path of the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Paths to files in the directories above
paths = [
    os.path.join(current_script_dir, '..', 'losses'),
    os.path.join(current_script_dir, '..', 'activations')
]

# Add all paths
sys.path += [os.path.dirname(name) for name in paths]

# Import files for testing
from n2rv.losses import loss_functions as lf

np.random.seed(0)


class TestLosses(unittest.TestCase):
    """Tests for loss functions."""

    def test_mse(self):
        """
        Test - compare outputs of Mean Squared Error loss function to
        manually calculated values.
        """
        y_true = np.array([
            [0.6],
            [1.2],
            [-0.8],
            [0.0],
            [-1.2]
        ])

        y_pred = np.array([
            [-0.1],
            [0.9],
            [-0.2],
            [0.2],
            [-1.1]
        ])

        desired = 0.198
        actual = lf.mse(y_pred=y_pred, y_true=y_true)

        self.assertAlmostEqual(actual, desired)

    def test_mse_derivative(self):
        """
        Test - compare outputs of Mean Squared Error loss derivative function
        to manually calculated values.
        """
        y_true = np.array([
            [0.6],
            [1.2],
            [-0.8],
            [0.0],
            [-1.2]
        ])

        y_pred = np.array([
            [-0.1],
            [0.9],
            [-0.2],
            [0.2],
            [-1.1]
        ])

        desired = np.array([
            [-0.28],
            [-0.12],
            [0.24],
            [0.08],
            [0.04]
        ])

        actual = lf.mse_derivative(y_pred=y_pred, y_true=y_true)

        np.testing.assert_allclose(actual=actual, desired=desired)

    def test_cce_with_softmax(self):
        """
        Test - compare outputs of Categorical Cross Entropy loss function to
        manually calculated values.
        """
        y_true = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ])

        y_pred = np.array([
            [0.1, 0.8, 0.1],
            [0.6, 0.3, 0.1],
            [0.5, 0.4, 0.1],
            [0.7, 0.2, 0.1],
            [0.7, 0.3, 0.0]
        ])

        desired = 0.9194404033
        actual = lf.categorical_cross_entropy_with_softmax(
            y_pred=y_pred,
            y_true=y_true
        )

        self.assertAlmostEqual(actual, desired)

    def test_cce_with_softmax_derivative(self):
        """
        Test - compare outputs of Categorical Cross Entropy loss derivative
        function to manually calculated values.
        """
        y_true = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ])

        y_pred = np.array([
            [0.1, 0.8, 0.1],
            [0.6, 0.3, 0.1],
            [0.5, 0.4, 0.1],
            [0.7, 0.2, 0.1],
            [0.7, 0.3, 0.0]
        ])

        desired = np.array([
            [0.1, -0.2, 0.1],
            [-0.4, 0.3, 0.1],
            [0.5, 0.4, -0.9],
            [-0.3, 0.2, 0.1],
            [0.7, -0.7, 0.0]
        ])

        actual = lf.categorical_cross_entropy_with_softmax_derivative(
            y_pred=y_pred,
            y_true=y_true
        )

        np.testing.assert_allclose(actual=actual, desired=desired)

    def test_bce(self):
        """
        Test - compare outputs of Binary Cross Entropy loss function to
        manually calculated values.
        """
        y_true = np.array([
            [0],
            [1],
            [0],
            [1],
            [0]
        ])

        y_pred = np.array([
            [0.1],
            [0.6],
            [0.5],
            [0.7],
            [0.7]
        ])

        desired = 0.5739962136
        actual = lf.binary_cross_entropy(y_pred=y_pred, y_true=y_true)

        self.assertAlmostEqual(desired, actual)

    def test_bce_derivative(self):
        """
        Test - compare outputs of Binary Cross Entropy loss derivative
        function to manually calculated values.
        """
        y_true = np.array([
            [0],
            [1],
            [0],
            [1],
            [0]
        ])

        y_pred = np.array([
            [0.1],
            [0.6],
            [0.5],
            [0.7],
            [0.7]
        ])

        desired = np.array([
            [1.111111111],
            [-1.666666667],
            [2.0],
            [-1.428571429],
            [3.333333333]
        ])

        actual = lf.binary_cross_entropy_derivative(
            y_pred=y_pred,
            y_true=y_true
        )

        np.testing.assert_allclose(actual=actual, desired=desired)

    def test_get_loss_correct_function_name(self):
        """
        Test - Requesting a loss function with correct name doesn't throw
        an exception.
        """
        try:
            lf.get_loss('mse')
        except ValueError:
            self.fail("loss_functions.get_loss() should not throw" +
                      " a ValueError with argument: mse")
        try:
            lf.get_loss('categorical_cross_entropy')
        except ValueError:
            self.fail("loss_functions.get_loss() should not throw" +
                      " a ValueError with argument: categorical cross entropy")

    def test_get_function_incorrect_function_name(self):
        """
        Test - Requesting a loss function with incorrect name throws an
        exception.
        """
        try:
            lf.get_loss('relu')
            self.fail("loss_functions.get_loss() should not allow" +
                      " argument: 'relu'")
        except ValueError:
            pass
        try:
            lf.get_loss('')
            self.fail("loss_functions.get_loss() should not allow" +
                      " empty string argument.")
        except ValueError:
            pass
        try:
            lf.get_loss(None)
            self.fail("loss_functions.get_loss() should not allow" +
                      " None argument")
        except ValueError:
            pass


if __name__ == '__main__':
    unittest.main(verbosity=2)
