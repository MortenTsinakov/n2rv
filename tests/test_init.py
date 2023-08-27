import unittest
import os
import sys
import numpy as np

# Get the absolute path of the current script (file_b.py)
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


class TestInit(unittest.TestCase):
    # This test passes
    def test_mse_passes(self):
        y_true = np.array([-1.0, 1.0, 2.0])
        y_pred = np.array([1.0, -2.0, 2.0])
        result = loss_functions.mse(y_true, y_pred)
        self.assertAlmostEqual(result, 4.3333333)

    # This test also passes
    def test_relu_passes(self):
        inputs = np.array([-1.0, 0.1, 1.0])
        result = activation_functions.relu(inputs)
        self.assertEqual(result.all(), np.array([0.0, 0.1, 1.0]).all())

    # This test fails
    def test_relu_fails(self):
        inputs = np.array([-1.0, 0.1, 1.0])
        result = activation_functions.relu(inputs)
        self.assertEqual(result.all(), np.array([1.0, 1.0, 1.0]).all())


if __name__ == '__main__':
    unittest.main()
