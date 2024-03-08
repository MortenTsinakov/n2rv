"""Tests for weight initialization."""


import unittest
import os
import sys
import numpy as np

# Get the absolute path of the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Paths to files in the directories above
paths = [
    os.path.join(current_script_dir, '..', 'weight_initialization')
]

# Add all paths
sys.path += [os.path.dirname(name) for name in paths]

# Import files for testing
from n2rv.weight_initialization import weight_initialization as w_i
np.random.seed(0)


class TestWeightInitialization(unittest.TestCase):
    """Tests for weight initialization."""
    def test_get_function_fan_in_none_returns_fan_out_sized_vector(self):
        """Test - Requesting weights with fan_in = None returns a vector of size fan_out."""
        fan_in = None
        fan_out = 4
        weights = w_i.get_weights(fan_in, fan_out)
        self.assertEqual(weights.shape, (fan_out, ))

    def test_get_function_fan_out_none_raises_exception(self):
        """Test - Requesting weights with fan_out = None throws an exception."""
        fan_in = 4
        fan_out = None
        try:
            w_i.get_weights(fan_in, fan_out)
            self.fail(
                "When fan_out is None, an exception should be raised when " +
                "calling weight_initialization.get_weights() function."
            )
        except TypeError:
            pass

    def test_get_function_technique_does_not_excist_throws_an_exception(self):
        """Test - Requesting weights with non-excisting technique throws an exception."""
        fan_in = 4
        fan_out = 2
        technique = "non_excisting_technique"
        try:
            w_i.get_weights(fan_in, fan_out, technique)
            self.fail(
                "Non-excisting technique should throw and exception in " +
                "weight_initialization.get_weights()."
            )
        except ValueError:
            pass

    def test_get_function_fan_in_non_positive_integer_throws_exception(self):
        """Test - Requesting weights with fan_in <= 0 throws an exception."""
        fail_text = "If fan_in is a non-positive integer when calling " +\
                    "weight_initialization.get_weights(), an exception " +\
                    "should be thrown."
        try:
            w_i.get_weights(fan_in=-1, fan_out=2)
            self.fail(fail_text)
        except ValueError:
            pass

        try:
            w_i.get_weights(fan_in=0, fan_out=2)
            self.fail(fail_text)
        except ValueError:
            pass

    def test_get_function_fan_out_non_positive_integer_throws_exception(self):
        """Test - Requesting weights with fan_out <= 0 throws an exception."""
        fail_text = "If fan_out is a non-positive integer when calling " +\
                    "weight_initialization.get_weights(), an exception " +\
                    "should be thrown."
        try:
            w_i.get_weights(fan_in=2, fan_out=-2)
            self.fail(fail_text)
        except ValueError:
            pass

        try:
            w_i.get_weights(fan_in=3, fan_out=0)
            self.fail(fail_text)
        except ValueError:
            pass

    def test_get_function_fan_out_non_integer_throws_exception(self):
        """Test - Requestin weights with fan_out of non-integer type throws an exception."""
        fail_text = "If fan_out is not an integer when calling " +\
                    "weight_initialization.get_weights() then " +\
                    "an exception should be thrown."
        try:
            w_i.get_weights(fan_in=2, fan_out='four')
            self.fail(fail_text)
        except TypeError:
            pass

        try:
            w_i.get_weights(fan_in=3, fan_out=0.5)
            self.fail(fail_text)
        except TypeError:
            pass

    def test_get_function_fan_in_non_integer_throws_exception(self):
        """Test - Requesting weights with fan_in of non-integer type throws an exception."""
        fail_text = "If fan_in is not an integer or None when calling " +\
                    "weight_initialization.get_weights() then " +\
                    "an exception should be thrown."
        try:
            w_i.get_weights(fan_in='two', fan_out=4)
            self.fail(fail_text)
        except TypeError:
            pass

        try:
            w_i.get_weights(fan_in=3.8, fan_out=8)
            self.fail(fail_text)
        except TypeError:
            pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
