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
    os.path.join(current_script_dir, '..', 'losses'),
    os.path.join(current_script_dir, '..', 'activations')
]

# Add all paths
sys.path += [os.path.dirname(name) for name in paths]

# Import files for testing
from activations import activation_functions
from losses import loss_functions

np.random.seed(0)


class TestLosses(unittest.TestCase):
    def test_mse(self):
        y_true = np.random.rand(10)
        y_pred = np.random.rand(10)

        tf_mse = tf.losses.mean_squared_error(y_true, y_pred).numpy()
        custom_mse = loss_functions.mse(y_true, y_pred)

        np.testing.assert_almost_equal(custom_mse, tf_mse, 1e-5)

    def test_categorical_cross_entropy(self):
        y_pred = activation_functions.softmax_with_categorical_cross_entropy(np.random.rand(5))
        y_true = np.array([0, 0, 1, 0, 0])

        tf_cce = tf.losses.categorical_crossentropy(y_true, y_pred).numpy()
        custom_cce = loss_functions.categorical_cross_entropy_with_softmax(y_true, y_pred)

        np.testing.assert_allclose(custom_cce, tf_cce, 1e-5)

    def test_mse_derivatives(self):
        y_true = np.random.rand(25)
        y_pred = np.random.rand(25)

        tf_y_true = tf.constant(y_true, dtype=tf.float32)
        tf_y_pred = tf.constant(y_pred, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(tf_y_true)
            tf_loss = tf.losses.mean_squared_error(tf_y_true, tf_y_pred)

        tf_grads = tape.gradient(tf_loss, tf_y_true).numpy()
        # Tensorflow calculates the MSE derivatives with the the opposite sign
        custom_grads = -loss_functions.mse_derivative(y_true, y_pred)

        np.testing.assert_allclose(custom_grads, tf_grads, 1e-5)

    def test_softmax_categorical_cross_entropy_derivatives(self):
        logits = np.random.rand(5)
        y_true = np.array([0, 0, 1, 0, 0])

        tf_logits = tf.constant(logits, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(tf_logits)
            tf_loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=tf_logits)

        tf_grads = tape.gradient(tf_loss, tf_logits).numpy()

        y_pred = activation_functions.softmax_with_categorical_cross_entropy(logits)
        custom_grads = loss_functions.categorical_cross_entropy_with_softmax_derivative(y_true, y_pred)

        np.testing.assert_allclose(custom_grads, tf_grads, 1e-5)

    def test_get_loss_correct_function_name(self):
        try:
            loss_functions.get_loss('mse')
        except ValueError:
            self.fail("loss_functions.get_loss() should not throw" +
                      " a ValueError with argument: mse")
        try:
            loss_functions.get_loss('categorical_cross_entropy')
        except ValueError:
            self.fail("loss_functions.get_loss() should not throw" +
                      " a ValueError with argument: categorical cross entropy")

    def test_get_function_incorrect_function_name(self):
        try:
            loss_functions.get_loss('relu')
            self.fail("loss_functions.get_loss() should not allow" +
                      " argument: 'relu'")
        except ValueError:
            pass
        try:
            loss_functions.get_loss('')
            self.fail("loss_functions.get_loss() should not allow" +
                      " empty string argument.")
        except ValueError:
            pass
        try:
            loss_functions.get_loss(None)
            self.fail("loss_functions.get_loss() should not allow" +
                      " None argument")
        except ValueError:
            pass


if __name__ == '__main__':
    unittest.main(verbosity=2)
