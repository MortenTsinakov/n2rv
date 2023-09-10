import unittest
import os
import sys

# Get the absolute path of the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Paths to files in the directories above
paths = [
    os.path.join(current_script_dir, '..', 'layers')
]

# Add all paths
sys.path += [os.path.dirname(name) for name in paths]

# Import files for testing
from layers import dense, input


class TestLayers(unittest.TestCase):
    # Dense layer initialization
    def test_dense_output_size_none_throws_exception(self):
        try:
            dense.Dense(output_size=None,
                        activation='relu')
            self.fail("Initializing Dense layer with output_size parameter " +
                      "as None should throw an exception.")
        except ValueError:
            pass

    def test_dense_output_size_less_than_one_throws_exception(self):
        try:
            dense.Dense(output_size=0,
                        activation='relu')
            self.fail("Initializing Dense layer with output_size parameter " +
                      "<0 should throw an exception.")
        except ValueError:
            pass
        try:
            dense.Dense(output_size=-4,
                        activation='relu')
            self.fail("Initializing Dense layer with output_size parameter " +
                      "<0 should throw an exception.")
        except ValueError:
            pass

    def test_dense_output_size_not_integer_throws_exception(self):
        try:
            dense.Dense(output_size=5.1,
                        activation='tanh')
            self.fail("Initializing Dense layer with output_size parameter " +
                      "as non-integer showld throw an exception.")
        except ValueError:
            pass
        try:
            dense.Dense(output_size="one",
                        activation='relu')
            self.fail("Initializing Dense layer with output_size parameter " +
                      "as non-integer showld throw an exception.")
        except ValueError:
            pass

    def test_dense_activation_none_throws_exception(self):
        try:
            dense.Dense(output_size=4,
                        activation=None)
            self.fail("Initializing Dense layer with activation parameter " +
                      "as None should throw an exception.")
        except ValueError:
            pass

    def test_dense_name_not_string_throws_exception(self):
        try:
            dense.Dense(output_size=4,
                        activation='tanh',
                        name=56)
            self.fail("Initializing Dense layer with name parameter " +
                      "as non-string should throw an exception.")
        except ValueError:
            pass

    # Dense __call__()
    def test_dense_call_invalid_previous_layer_throws_exception(self):
        try:
            dense.Dense(output_size=4,
                        activation='tanh')('layer')
            self.fail("Calling Dense layer with non-Layer type argument " +
                      "should throw an exception.")
        except ValueError:
            pass
        try:
            dense.Dense(output_size=5,
                        activation='sigmoid')(None)
            self.fail("Calling Dense layer with non-Layer type argument " +
                      "should throw an exception.")
        except ValueError:
            pass
        try:
            dense.Dense(output_size=2,
                        activation='relu')(24)
            self.fail("Calling Dense layer with non-Layer type argument " +
                      "should throw an exception.")
        except ValueError:
            pass

    # Input initialization
    def test_input_shape_type_not_tuple_throws_exception(self):
        try:
            input.Input(shape=14)
            self.fail("Input shape of non-tuple type should throw an " +
                      "exception.")
        except TypeError:
            pass
        try:
            input.Input(shape=None)
            self.fail("Input shape of non-tuple type should throw an " +
                      "exception.")
        except TypeError:
            pass
        try:
            input.Input(shape=[2, 3, 4])
            self.fail("Input shape of non-tuple type should throw an " +
                      "exception.")
        except TypeError:
            pass
        try:
            input.Input(shape="tuple")
            self.fail("Input shape of non-tuple type should throw an " +
                      "exception.")
        except TypeError:
            pass

    def test_input_shape_elements_leq_zero_throws_exception(self):
        try:
            input.Input(shape=(-6, 2))
            self.fail("Input shape consisting of non-positive integers " +
                      "should throw an exception.")
        except ValueError:
            pass
        try:
            input.Input(shape=(0, 4))
            self.fail("Input shape consisting of non-positive integers " +
                      "should throw an exception.")
        except ValueError:
            pass
        try:
            input.Input(shape=(4, 0))
            self.fail("Input shape consisting of non-positive integers " +
                      "should throw an exception.")
        except ValueError:
            pass
        try:
            input.Input(shape=(0, 0))
            self.fail("Input shape consisting of non-positive integers " +
                      "should throw an exception.")
        except ValueError:
            pass
        try:
            input.Input(shape=(2, -1))
            self.fail("Input shape consisting of non-positive integers " +
                      "should throw an exception.")
        except ValueError:
            pass

    def test_input_shape_empty_tuple_throws_exception(self):
        try:
            input.Input(shape=())
            self.fail("Input shape of empty tuple should throw an exception.")
        except ValueError:
            pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
