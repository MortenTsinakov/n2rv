"""Tests for Model class."""


import unittest
import os
import sys
import numpy as np

# Get the absolute path of the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Paths to files in the directories above
paths = [
    os.path.join(current_script_dir, '..', 'models'),
    os.path.join(current_script_dir, '..', 'layers'),
    os.path.join(current_script_dir, '..', 'exceptions'),
    os.path.join(current_script_dir, '..', 'optimizers'),
]

# Add all paths
sys.path += [os.path.dirname(name) for name in paths]

# Import files for testing
from models.model import Model
from layers.dense import Dense
from layers.input import Input
from optimizers.sgd import SGD
from exceptions.exception import IncompatibleLayerError,\
                                 DisconnectedLayersError,\
                                 ShapeMismatchError

np.random.seed(0)


class TestModel(unittest.TestCase):
    """Tests for Model class."""
    def test_model_init_no_inputs_throws_exception(self):
        """Test - initialize model with inputs as None"""
        try:
            outputs = Dense(output_size=4, activation='relu')
            Model(inputs=None, outputs=outputs)
            self.fail("Initializing Model with no inputs should throw " +
                      "an exception.")
        except ValueError:
            pass

    def test_model_init_no_outputs_throws_exception(self):
        """Test - initialize model with outputs as None."""
        try:
            inputs = Input(shape=(2, 3))
            Model(inputs=inputs, outputs=None)
            self.fail("Initializing Model with no outputs should " +
                      "throw and exception.")
        except ValueError:
            pass

    def test_model_compile_loss_fn_none_throws_exception(self):
        """Test - Compile model with no loss function."""
        try:
            inputs = Input((2, 3))
            outputs = Dense(output_size=1, activation='relu')(inputs)
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(loss_fn=None, optimizer=SGD())
            self.fail("None as loss function of Model.compile(), should " +
                      "throw an exception.")
        except ValueError:
            pass

    # Softmax activation function should only be used with
    # Categorical Cross Entropy loss.
    def test_model_compile_using_softmax_only_throws_exception(self):
        """Test - use Softmax activation function without CCE loss."""
        try:
            inputs = Input(shape=(2, ))
            outputs = Dense(output_size=1, activation='softmax')(inputs)
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(loss_fn='mse', optimizer=SGD())
            self.fail("Using softmax without CCE loss should throw an " +
                      "exception.")
        except IncompatibleLayerError:
            pass

    # Categorical cross-entropy loss should only be used with 
    # softmax activation function.
    def test_model_compile_using_cce_only_throws_exception(self):
        """Test - use Categorical Cross-Entropy loss without Softmax activation function."""
        try:
            inputs = Input(shape=(3, ))
            outputs = Dense(output_size=1, activation='tanh')(inputs)
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(loss_fn='categorical_cross_entropy', optimizer=SGD())
            self.fail("Using Categorical Cross-Entropy without Softmax " +
                      "should throw and exception.")
        except IncompatibleLayerError:
            pass

    # Softmax activation function should only be used as the activation
    # function in the last layer.
    def test_model_compile_softmax_mid_model_throws_exception(self):
        """Test - use Softmax function in a layer that is not the last."""
        try:
            inputs = Input(shape=(2, ))
            x = Dense(output_size=8, activation='softmax')(inputs)
            outputs = Dense(output_size=1, activation='relu')(x)
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(loss_fn='categorical_cross_entropy', optimizer=SGD())
        except IncompatibleLayerError:
            pass

    def test_model_compile_inputs_outputs_not_connected_throws_exception(self):
        """Test - compile the model without a connection from inputs to outputs."""
        try:
            inputs = Input(shape=(2, ))
            outputs = Dense(output_size=4, activation='relu')
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(loss_fn='mse', optimizer=SGD())
            self.fail("No connection between inputs-outputs should throw " +
                      "an exception.")
        except DisconnectedLayersError:
            pass

    def test_model_predict_wrong_x_shape_throws_exception(self):
        """Test - feed data with wrong shape into the model on prediction."""
        try:
            inputs = Input(shape=(2, ))
            outputs = Dense(output_size=1, activation='linear')(inputs)
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(loss_fn='mse', optimizer=SGD())

            x = np.random.rand(4, 3)

            model.predict(x)
            self.fail("Mismatch between Input layer shape and data shape " +
                      "fed into model.predict() should throw an exception.")
        except ShapeMismatchError:
            pass

    def test_model_predict_wrong_x_type_throws_exception(self):
        """Test - feed None into the model on prediction."""
        inputs = Input(shape=(2, ))
        outputs = Dense(output_size=1, activation='linear')(inputs)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss_fn='mse', optimizer=SGD())
        error_text = "Feeding anything other than Numpy array into " +\
                     "Model.predict() should throw and exception."
        try:
            x = None
            model.predict(x)
            self.fail(error_text)
        except TypeError:
            pass
        try:
            x = [[1, 2], [2, 3], [2, 1], [0, 0]]
            model.predict(x)
            self.fail(error_text)
        except TypeError:
            pass
        try:
            x = 'hello!'
            model.predict(x)
            self.fail(error_text)
        except TypeError:
            pass

    def test_model_fit_wrong_x_type_throws_exception(self):
        """Test - feed None into the model on training."""
        inputs = Input(shape=(2, ))
        outputs = Dense(output_size=1, activation='linear')(inputs)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss_fn='mse', optimizer=SGD())
        error_text = "Feeding anything other than Numpy array into " +\
                     "Model.fit() should throw and exception."
        try:
            x = None
            y = np.random.rand(4, 1)

            model.fit(x_train=x,
                      y_train=y,
                      epochs=1,
                      print_metrics=False,
                      batch_size=1)
            self.fail(error_text)
        except TypeError:
            pass
        try:
            x = [1, 2, 3]
            y = np.random.rand(4, 1)

            model.fit(x_train=x,
                      y_train=y,
                      epochs=1,
                      print_metrics=False,
                      batch_size=1)
            self.fail(error_text)
        except TypeError:
            pass
        try:
            x = 'hello!'
            y = np.random.rand(4, 1)

            model.fit(x_train=x,
                      y_train=y,
                      epochs=1,
                      print_metrics=False,
                      batch_size=1)
            self.fail(error_text)
        except TypeError:
            pass

    def test_model_fit_wrong_x_shape_throws_exception(self):
        """Test - feed data with wrong shape into the model on training."""
        try:
            inputs = Input(shape=(2, ))
            outputs = Dense(output_size=1, activation='linear')(inputs)
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(loss_fn='mse', optimizer=SGD())

            x = np.random.rand(4, 3)
            y = np.random.rand(4, 1)

            model.fit(x_train=x,
                      y_train=y,
                      epochs=1,
                      print_metrics=False,
                      batch_size=1)
            self.fail("Mismatch between Input layer shape and data shape " +
                      "fed into model.fit() should throw an exception.")
        except ShapeMismatchError:
            pass

    def test_model_fit_wrong_y_type_throws_exception(self):
        """Test - feed None as labels in to the model on training."""
        inputs = Input(shape=(2, ))
        outputs = Dense(output_size=1, activation='linear')(inputs)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss_fn='mse', optimizer=SGD())
        error_text = "Feeding anything other than Numpy array into " +\
                     "Model.fit() should throw an exception."
        try:
            x = np.random.rand(4, 2)
            y = None

            model.fit(x_train=x,
                      y_train=y,
                      epochs=1,
                      print_metrics=False,
                      batch_size=1)
            self.fail(error_text)
        except TypeError:
            pass
        try:
            x = np.random.rand(4, 2)
            y = [1, 2, 3, 4]

            model.fit(x_train=x,
                      y_train=y,
                      epochs=1,
                      print_metrics=False,
                      batch_size=1)
            self.fail(error_text)
        except TypeError:
            pass
        try:
            x = np.random.rand(4, 2)
            y = 'hello'

            model.fit(x_train=x,
                      y_train=y,
                      epochs=1,
                      print_metrics=False,
                      batch_size=1)
            self.fail(error_text)
        except TypeError:
            pass

    def test_model_fit_wrong_y_shape_throws_exception(self):
        """Test - feed wrong shaped labels into the model on training."""
        inputs = Input(shape=(2, ))
        outputs = Dense(output_size=1, activation='linear')(inputs)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss_fn='mse', optimizer=SGD())
        try:
            x = np.random.rand(15, 2)
            y = np.random.rand(15, 2)
            model.fit(x_train=x,
                      y_train=y,
                      epochs=1,
                      print_metrics=False,
                      batch_size=1)
            self.fail("An exception should be thrown if the output layer's " +
                      "output dimension doesn't match the label dimensions.")
        except ShapeMismatchError:
            pass

    def test_model_fit_different_length_x_y_throws_exception(self):
        """Test - feed different number of samples and labels into the model on training."""
        inputs = Input(shape=(2, ))
        outputs = Dense(output_size=1, activation='linear')(inputs)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss_fn='mse', optimizer=SGD())
        try:
            x = np.random.rand(15, 2)
            y = np.random.rand(13, 1)
            model.fit(x_train=x,
                      y_train=y,
                      epochs=1,
                      print_metrics=False,
                      batch_size=1)
            self.fail("If the number of examples doesn't fit the number of labesl " +
                      "an exception should be thrown.")
        except ShapeMismatchError:
            pass

    def test_model_fit_incorrect_epochs_argument_throws_exception(self):
        """Test - pass invalid epochs argument when training the model."""
        inputs = Input(shape=(2, ))
        outputs = Dense(output_size=1, activation='linear')(inputs)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss_fn='mse', optimizer=SGD())
        x = np.random.rand(4, 2)
        y = np.random.rand(4, 1)
        value_error_text = "Passing epoch argument that is <= 0 into " +\
                           " Model.fit() should throw an exception."
        type_error_text = "Passing an epoch argument of any other type " +\
                          "than integer should throw an exception."
        try:
            epochs = 0
            model.fit(x_train=x,
                      y_train=y,
                      epochs=epochs,
                      print_metrics=False,
                      batch_size=1)
            self.fail(value_error_text)
        except ValueError:
            pass
        try:
            epochs = -10
            model.fit(x_train=x,
                      y_train=y,
                      epochs=epochs,
                      print_metrics=False,
                      batch_size=1)
            self.fail(value_error_text)
        except ValueError:
            pass
        try:
            epochs = None
            model.fit(x_train=x,
                      y_train=y,
                      epochs=epochs,
                      print_metrics=False,
                      batch_size=1)
            self.fail(type_error_text)
        except TypeError:
            pass
        try:
            epochs = 'thousand'
            model.fit(x_train=x,
                      y_train=y,
                      epochs=epochs,
                      print_metrics=False,
                      batch_size=1)
            self.fail(type_error_text)
        except TypeError:
            pass
        try:
            epochs = 0.2
            model.fit(x_train=x,
                      y_train=y,
                      epochs=epochs,
                      print_metrics=False,
                      batch_size=1)
            self.fail(type_error_text)
        except TypeError:
            pass

    def test_model_fit_incorrect_print_loss_argument_throws_exception(self):
        """Test - pass incorrect print_loss argument while training the model."""
        inputs = Input(shape=(2, ))
        outputs = Dense(output_size=1, activation='linear')(inputs)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss_fn='mse', optimizer=SGD())
        x = np.random.rand(4, 2)
        y = np.random.rand(4, 1)
        type_error_text = "Passing a non-Boolean print_loss argument to " +\
                          "Model.fit() should throw an exception."
        try:
            print_loss = 4
            model.fit(x_train=x,
                      y_train=y,
                      epochs=1,
                      print_metrics=print_loss,
                      batch_size=1)
            self.fail(type_error_text)
        except TypeError:
            pass
        try:
            print_loss = "no"
            model.fit(x_train=x,
                      y_train=y,
                      epochs=1,
                      print_metrics=print_loss,
                      batch_size=1)
            self.fail(type_error_text)
        except TypeError:
            pass

    def test_model_fit_incorrect_batch_size_argument_throws_exception(self):
        """Test - pass incorrect batch_size argument into the model when training."""
        inputs = Input(shape=(2, ))
        outputs = Dense(output_size=1, activation='linear')(inputs)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss_fn='mse', optimizer=SGD())
        x = np.random.rand(4, 2)
        y = np.random.rand(4, 1)
        type_error_text = "Passing a non-integer type batch_size argument " +\
                          "to Model.fit() should throw an exception."
        value_error_text = "Passing a batch_size argument <0 or " +\
                           ">len(x_train) should throw an exception."
        try:
            batch_size = None
            model.fit(x_train=x,
                      y_train=y,
                      epochs=1,
                      print_metrics=False,
                      batch_size=batch_size)
            self.fail(type_error_text)
        except TypeError:
            pass
        try:
            batch_size = 0.3
            model.fit(x_train=x,
                      y_train=y,
                      epochs=1,
                      print_metrics=False,
                      batch_size=batch_size)
            self.fail(type_error_text)
        except TypeError:
            pass
        try:
            batch_size = "big"
            model.fit(x_train=x,
                      y_train=y,
                      epochs=1,
                      print_metrics=False,
                      batch_size=batch_size)
            self.fail(type_error_text)
        except TypeError:
            pass
        try:
            batch_size = -2
            model.fit(x_train=x,
                      y_train=y,
                      epochs=1,
                      print_metrics=False,
                      batch_size=batch_size)
            self.fail(value_error_text)
        except ValueError:
            pass
        try:
            batch_size = 0
            model.fit(x_train=x,
                      y_train=y,
                      epochs=1,
                      print_metrics=False,
                      batch_size=batch_size)
            self.fail(value_error_text)
        except ValueError:
            pass
        try:
            batch_size = len(x) + 1
            model.fit(x_train=x,
                      y_train=y,
                      epochs=1,
                      print_metrics=False,
                      batch_size=batch_size)
            self.fail(value_error_text)
        except ValueError:
            pass


if __name__ == '__main__':
    unittest.main(verbosity=2)
