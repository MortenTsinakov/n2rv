"""Test for metrics."""


import unittest
import os
import sys
import numpy as np

# Get the absolute path of the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Paths to files in the directories above
paths = [
    os.path.join(current_script_dir, '..', 'metrics')
]

# Add all paths
sys.path += [os.path.dirname(name) for name in paths]

# Import files for testing
from n2rv.metrics.accuracy import Accuracy
from n2rv.metrics.binary_accuracy import BinaryAccuracy
from n2rv.metrics.mse import MSE
from n2rv.metrics.mae import MAE
from n2rv.metrics.precision import Precision
from n2rv.metrics.recall import Recall
from n2rv.metrics.model_metrics import ModelMetrics
from n2rv.exceptions.exception import DuplicateItemsError, ShapeMismatchError


class TestMetrics(unittest.TestCase):
    """Tests for Metrics."""

    def test_model_metrics_input_none_metrics_list_is_empty(self):
        """
        Test - ModelMetrics metrics is an empty list if initialized
        with None value.
        """
        m = ModelMetrics(metrics_list=None)
        assert m.metrics_list == []

    def test_model_metrics_metrics_list_wrong_type(self):
        """
        Test - ModelMetrics metrics list not of type list
        raises an exception.
        """
        try:
            ModelMetrics(metrics_list="[]")
            self.fail("Metrics_list of wrong type on ModelMetrics" +
                      "initialization should raise an exception.")
        except TypeError:
            pass

        try:
            ModelMetrics(metrics_list=set())
            self.fail("Metrics_list of wrong type on ModelMetrics" +
                      "initialization should raise an exception.")
        except TypeError:
            pass

        try:
            ModelMetrics(metrics_list={})
            self.fail("Metrics_list of wrong type on ModelMetrics" +
                      "initialization should raise an exception.")
        except TypeError:
            pass

        try:
            ModelMetrics(metrics_list=Accuracy())
            self.fail("Metrics_list of wrong type on ModelMetrics" +
                      "initialization should raise an exception.")
        except TypeError:
            pass

    def test_model_metrics_metrics_not_of_type_metrics(self):
        """
        Test - Metrics list containing items that are not of Metrics
        class raises an exception.
        """
        try:
            ModelMetrics(metrics_list=[5, 4, 1])
            self.fail("Metrics_list containing an object that is not from" +
                      "Metrics class should raise an exception.")
        except TypeError:
            pass

        try:
            ModelMetrics(metrics_list=[[Accuracy()]])
            self.fail("Metrics_list containing an object that is not from" +
                      "Metrics class should raise an exception.")
        except TypeError:
            pass

        try:
            ModelMetrics(metrics_list=[Accuracy, 5])
            self.fail("Metrics_list containing an object that is not from" +
                      "Metrics class should raise an exception.")
        except TypeError:
            pass

    def test_model_metrics_metrics_list_contains_duplicate_metrics(self):
        """
        Test - Metrics list containing duplicate metrics on ModelMetrics
        initialization should raise an exception.
        """
        try:
            ModelMetrics(metrics_list=[Accuracy(), Accuracy()])
            self.fail("Metrics list containing duplicate metrics should" +
                      "raise an exception.")
        except DuplicateItemsError:
            pass

    def test_model_metrics_shape_mismatch_on_update(self):
        """
        Test - Updating metrics with true and predicted values of different
        shape raises an exception.
        """
        mm = ModelMetrics(metrics_list=[BinaryAccuracy()])
        true = np.array([
            [1],
            [1],
            [0]
        ])
        pred = np.array([
            [0.1],
            [0.9]
        ])
        with self.assertRaises(ShapeMismatchError):
            mm.update_metrics(true=true, pred=pred)

    def test_model_metrics_updates_metrics(self):
        """Test - ModelMetrics correctly updates metrics."""
        a = Accuracy()
        m = ModelMetrics(metrics_list=[a])

        true = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 1]
        ])
        pred = np.array([
            [0.8, 0.1, 0.1],    # Correct prediction
            [0.1, 0.6, 0.3],    # Correct prediction
            [0.7, 0.1, 0.2],    # Incorrect prediction
            [0.3, 0.2, 0.5],    # Incorrect prediction
            [0.3, 0.1, 0.6]     # Correct prediction
        ])

        m.update_metrics(true=true, pred=pred)
        assert a.get_metric() == 0.6

    def test_model_metrics_resets_metrics(self):
        """Tets - ModelMetrics resets metrics for Metric in the list."""
        a = Accuracy()
        m = ModelMetrics(metrics_list=[a])

        m.reset_metrics()
        assert a.get_metric() == 0

    def test_accuracy_initialization_input_wrong_type(self):
        """
        Test - Input of wrong type raises exception on Accuracy
        initialization.
        """
        decimal_places_str = "0.5"
        decimal_places_float = 0.5
        decimal_places_list = ["0.5"]
        
        try:
            Accuracy(decimal_places=decimal_places_str)
            self.fail("Input of wrong type should raise an exception" +
                      "on Accuracy initialization.")
        except TypeError:
            pass

        try:
            Accuracy(decimal_places=decimal_places_float)
            self.fail("Input of wrong type should raise an exception" +
                      "on Accuracy initialization.")
        except TypeError:
            pass

        try:
            Accuracy(decimal_places=decimal_places_list)
            self.fail("Input of wrong type should raise an exception" +
                      "on Accuracy initialization.")
        except TypeError:
            pass

    def test_accuracy_initializaiton_input_value_out_of_bounds(self):
        """
        Test - Input value not in range [0, 10] raises Exception
        on Accuracy initialization.
        """
        try:
            Accuracy(decimal_places=-1)
            self.fail("Negative value for decimal places on Accuracy" +
                      "initialization should raise an exception.")
        except ValueError:
            pass

        try:
            Accuracy(decimal_places=11)
            self.fail("Decimal places with value >10 should raise an exception.")
        except ValueError:
            pass

    def test_accuracy_get_before_updates(self):
        """Test - Method get_metric returns 0 before updates."""
        a = Accuracy()
        assert a.get_metric() == 0

    def test_accuracy_get_after_update(self):
        """Test - Method get_metric returns correct value after update."""
        a = Accuracy()
        true = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 1]
        ])
        pred = np.array([
            [0.8, 0.1, 0.1],    # Correct prediction
            [0.1, 0.6, 0.3],    # Correct prediction
            [0.7, 0.1, 0.2],    # Incorrect prediction
            [0.3, 0.2, 0.5],    # Incorrect prediction
            [0.3, 0.1, 0.6]     # Correct prediction
        ])

        a.update(true=true, pred=pred)
        assert a.get_metric() == 0.6

    def test_accuracy_update_several_times_before_get(self):
        """
        Test - Method get_metric returns correct result after
        several updates.
        """
        a = Accuracy()
        true = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 1]
        ])
        pred = np.array([
            [0.8, 0.1, 0.1],    # Correct prediction
            [0.1, 0.6, 0.3],    # Correct prediction
            [0.7, 0.1, 0.2],    # Incorrect prediction
            [0.3, 0.2, 0.5],    # Incorrect prediction
            [0.3, 0.1, 0.6]     # Correct prediction
        ])

        true2 = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 1]
        ])
        pred2 = np.array([
            [0.1, 0.1, 0.8],    # Incorrect prediction
            [0.1, 0.8, 0.1],    # Correct prediction
            [0.7, 0.1, 0.2],    # Incorrect prediction
            [0.3, 0.2, 0.5],    # Incorrect prediction
            [0.7, 0.1, 0.2]     # Inorrect prediction
        ])

        a.update(true=true, pred=pred)
        a.update(true=true2, pred=pred2)
        assert a.get_metric() == 0.4

    def test_accuracy_update_reset_update(self):
        """Test - Method get_metric returns correct result after reset."""
        a = Accuracy()
        true = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 1]
        ])
        pred = np.array([
            [0.8, 0.1, 0.1],    # Correct prediction
            [0.1, 0.6, 0.3],    # Correct prediction
            [0.7, 0.1, 0.2],    # Incorrect prediction
            [0.3, 0.2, 0.5],    # Incorrect prediction
            [0.3, 0.1, 0.6]     # Correct prediction
        ])

        a.update(true=true, pred=pred)
        assert a.get_metric() == 0.6

        true2 = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 1]
        ])
        pred2 = np.array([
            [0.1, 0.1, 0.8],    # Incorrect prediction
            [0.1, 0.8, 0.1],    # Correct prediction
            [0.7, 0.1, 0.2],    # Incorrect prediction
            [0.3, 0.2, 0.5],    # Incorrect prediction
            [0.7, 0.1, 0.2]     # Inorrect prediction
        ])

        a.reset()
        a.update(true=true2, pred=pred2)
        assert a.get_metric() == 0.2

    def test_binary_accuracy_wrong_decimal_places_type(self):
        """
        Test - initializing Binary Accuracy with wrong decimal places
        argument type raises an exception.
        """
        with self.assertRaises(TypeError):
            BinaryAccuracy(decimal_places=0.6)
        with self.assertRaises(TypeError):
            BinaryAccuracy(decimal_places=[5])
        with self.assertRaises(TypeError):
            BinaryAccuracy(decimal_places="8")

    def test_binary_accuracy_decimal_places_is_of_wrong_value(self):
        """
        Test - initializing Binary Accuracy with decimal places < 0 or
        >10 raises an exception.
        """
        with self.assertRaises(ValueError):
            BinaryAccuracy(decimal_places=-2)
        with self.assertRaises(ValueError):
            BinaryAccuracy(decimal_places=11)

    def test_binary_accuracy_threshold_wrong_type(self):
        """
        Test - initializing Binary Accuracy with wrong threshold argument
        type raises an exception.
        """
        with self.assertRaises(TypeError):
            BinaryAccuracy(threshold=2)
        with self.assertRaises(TypeError):
            BinaryAccuracy(threshold="0.5")
        with self.assertRaises(TypeError):
            BinaryAccuracy(threshold=[0.1])

    def test_binary_accuracy_returns_zero_if_no_updates(self):
        """
        Test - Binary Accuracy returns 0 if no updates have been
        performed.
        """
        bin_acc = BinaryAccuracy()
        self.assertEqual(bin_acc.get_metric(), 0)

    def test_binary_accuracy_update(self):
        """
        Test - Updating Binary Accuracy works.
        """
        true = np.array([
            [0],
            [1],
            [0]
        ])
        pred = np.array([
            [0.2],
            [0.8],
            [0.7]
        ])
        bin_acc = BinaryAccuracy(decimal_places=10)
        bin_acc.update(true=true, pred=pred)

        expected = 0.6666666667
        actual = bin_acc.get_metric()
        self.assertAlmostEqual(actual, expected)

    def test_binary_accuracy_update_several_times(self):
        """
        Test - Updating Binary Accuracy several times before
        getting metrics works correctly.
        """
        true = np.array([
            [1],
            [1],
            [0],
            [1],
            [0]
        ])
        pred = np.array([
            [0.2],
            [0.8],
            [0.1],
            [0.6],
            [0.7]
        ])
        bin_acc = BinaryAccuracy(decimal_places=10)
        bin_acc.update(true=true[:3], pred=pred[:3])
        bin_acc.update(true=true[3:], pred=pred[3:])

        expected = 0.6
        actual = bin_acc.get_metric()

        self.assertAlmostEqual(actual, expected)

    def test_binary_accuracy_reset(self):
        """
        Test - Reseting Binary Accuracy works.
        """
        bin_acc = BinaryAccuracy(decimal_places=10)
        true = np.array([
            [1],
            [1],
            [0]
        ])
        pred = np.array([
            [0.6],
            [0.2],
            [0.1]
        ])
        bin_acc.update(true=true, pred=pred)
        bin_acc.reset()

        self.assertEqual(bin_acc.get_metric(), 0)

    def test_binary_accuracy_update_reset_update(self):
        """
        Test - Update Binary Accuracy, the reset and update again
        works correctly.
        """
        bin_acc = BinaryAccuracy(decimal_places=10)
        true = np.array([
            [1],
            [1],
            [0]
        ])
        pred = np.array([
            [0.6],
            [0.2],
            [0.1]
        ])
        bin_acc.update(true=true, pred=pred)
        bin_acc.reset()
        bin_acc.update(true=true, pred=pred)

        expected = 2 / 3
        actual = bin_acc.get_metric()

        self.assertAlmostEqual(actual, expected)

    def test_mse_metrics_initialization_with_correct_decimal_places_succeeds(self):
        """
        Test - Initialize MSE metric with valid parameters.
        """
        try:
            for i in range(11):
                MSE(decimal_places=i)
        except Exception:
            self.fail("MSE metric should not raise an exception " +
                      "when initialized with valid arguments.")

    def test_mse_metrics_validation_decimal_places_incorrect_type(self):
        """
        Test - Initializing MSE metrics with arguments of wrong type should
        raise an exception.
        """
        with self.assertRaises(TypeError):
            MSE(decimal_places=1.2)
        with self.assertRaises(TypeError):
            MSE(decimal_places="5")
        with self.assertRaises(TypeError):
            MSE(decimal_places=[1])

    def test_mse_metrics_validation_decimal_places_incorrect_value(self):
        """
        Test - Initializing MSE metrics with arguments of incorrect value
        should raise an exception.
        """
        with self.assertRaises(ValueError):
            MSE(decimal_places=-1)
        with self.assertRaises(ValueError):
            MSE(decimal_places=11)

    def test_mse_metrics_update_once(self):
        """
        Test - Updating MSE metric once returns the correct value.
        """
        mse = MSE(decimal_places=10)
        true = np.array([
            0.2,
            0.1,
            0.5,
            0.3,
        ])
        pred = np.array([
            0.1,
            0.5,
            0.2,
            0.3,
        ])
        expected = 0.065

        mse.update(true=true, pred=pred)
        actual = mse.get_metric()
        self.assertAlmostEqual(expected, actual)

    def test_mse_metrics_update_several_times(self):
        """
        Test - Updating MSE metrics several times returns the
        correct value.
        """
        mse = MSE(decimal_places=10)

        true1 = np.array([0.2, -0.3, 0.1, 0.6])
        true2 = np.array([0.4, 0.2, -0.7, 0.5])
        true3 = np.array([-0.3, 0.8, 1.2, -0.9])

        pred1 = np.array([0.1, 0.2, -0.2, 0.6])
        pred2 = np.array([0.7, -0.5, 0.3, -0.6])
        pred3 = np.array([0.8, 0.2, -0.4, 0.5])

        mse.update(true=true1, pred=pred1)
        mse.update(true=true2, pred=pred2)
        mse.update(true=true3, pred=pred3)

        expected = 0.769_166_6667
        actual = mse.get_metric()

        self.assertAlmostEqual(expected, actual)

    def test_mse_metrics_returns_zero_after_reset(self):
        """
        Test - MSE metrics returns 0 after reset.
        """
        mse = MSE(decimal_places=10)

        true = np.array([0.4, -0.1, 0.8])
        pred = np.array([0.7, -0.4, 0.7])

        mse.update(true=true, pred=pred)
        mse.reset()

        self.assertEqual(0, mse.get_metric())

    def test_mse_metrics_returns_zero_before_update(self):
        """
        Test - MSE metrics returns 0 before update.
        """
        mse = MSE(decimal_places=0)

        self.assertEqual(0, mse.get_metric())

    def test_mae_metrics_correct_initialization(self):
        """
        Test - MAE initializes successfully if arguments are valid.
        """
        try:
            for i in range(11):
                MAE(decimal_places=i)
        except Exception:
            self.fail("Initializing MAE with correct arguments " +
                      "shouldn't raise an exception.")

    def test_mae_metrics_validation_arguments_of_wrong_type(self):
        """
        Test - Initializing MAE with arguments of wrong type raises
        TypeError.
        """
        with self.assertRaises(TypeError):
            MAE(decimal_places=1.2)
        with self.assertRaises(TypeError):
            MAE(decimal_places="5")
        with self.assertRaises(TypeError):
            MAE(decimal_places=[3])

    def test_mae_metrics_validation_aruments_out_of_range(self):
        """
        Test - Initializing MAE with arguments out of range raises
        an ValueError
        """
        with self.assertRaises(ValueError):
            MAE(decimal_places=-1)
        with self.assertRaises(ValueError):
            MAE(decimal_places=11)

    def test_mae_metrics_update_once(self):
        """
        Test - Updating MAE metrics once returns the correct value.
        """
        mae = MAE(decimal_places=10)

        true = np.array([0.5, 0.1, -0.8, 0.0])
        pred = np.array([0.4, -0.1, -0.4, 0.2])
        mae.update(true=true, pred=pred)

        expected = 0.225
        actual = mae.get_metric()

        self.assertAlmostEqual(expected, actual)

    def test_mae_metrics_update_several_times(self):
        """
        Test - Updating MAE metrics several times returns correct
        value.
        """
        mae = MAE(decimal_places=10)

        true1 = np.array([0.3, -0.2, 0.1])
        true2 = np.array([0.1, 0.2, -0.8])
        true3 = np.array([-0.3, 0.5, 0.2])

        pred1 = np.array([0.1, -0.4, 0.3])
        pred2 = np.array([0.5, -0.4, 0.2])
        pred3 = np.array([-0.1, 0.3, 0.0])

        mae.update(true=true1, pred=pred1)
        mae.update(true=true2, pred=pred2)
        mae.update(true=true3, pred=pred3)

        expected = 0.355_555_555_6
        actual = mae.get_metric()

        self.assertAlmostEqual(expected, actual)

    def test_mae_metrics_resets_correctly(self):
        """
        Test - MAE metrics resets to 0
        """
        mae = MAE(decimal_places=10)

        true = np.array([0.5, 0.1, -0.8, 0.0])
        pred = np.array([0.4, -0.1, -0.4, 0.2])
        mae.update(true=true, pred=pred)

        mae.reset()

        self.assertEqual(0, mae.get_metric())

    def test_mae_metrics_returns_zero_before_update(self):
        """
        Test - MAE metrics returns 0 before updates
        """
        mae = MAE()

        self.assertEqual(0, mae.get_metric())

    def test_precision_metrics_correct_initialization(self):
        """
        Test - Precision metrics initialization works with correct
        parameters
        """
        for i in range(11):
            Precision(decimal_places=i, threshold=0.8)

    def test_precision_metrics_decimal_places_incorrect_type_on_init(self):
        """
        Test - Initializing Precision with decimal plces of wrong type
        raises a TypeError.
        """
        with self.assertRaises(TypeError):
            Precision(decimal_places=2.5)
        with self.assertRaises(TypeError):
            Precision(decimal_places=[1])
        with self.assertRaises(TypeError):
            Precision("1")

    def test_precision_metrics_decimal_places_incorrect_value_on_init(self):
        """
        Test - Initializing Precision with decimal places out of range
        raises ValueError.
        """
        with self.assertRaises(ValueError):
            Precision(decimal_places=-1)
        with self.assertRaises(ValueError):
            Precision(decimal_places=11)
    
    def test_precision_metrics_threshold_incorrect_type_on_init(self):
        """
        Test - Initializing Precision with threshold of wrong type
        raises TypeError.
        """
        with self.assertRaises(TypeError):
            Precision(threshold=7)
        with self.assertRaises(TypeError):
            Precision(threshold="0.5")
        with self.assertRaises(TypeError):
            Precision(threshold=[0.5])

    def test_precision_metrics_counts_thresholds_correctly(self):
        """
        Test - Updating Precision with specific threshold works
        correctly.
        """
        precision = Precision(decimal_places=10,
                              threshold=0.8)
        true = np.array([1, 0, 0, 1, 1, 0, 1, 0])
        pred = np.array([0.9, 0.8, 0.7, 0.8, 0.1, 0.6, 1, 0.8])

        expected = 3 / (3 + 2)

        precision.update(true=true, pred=pred)
        actual = precision.get_metric()

        self.assertAlmostEqual(expected, actual)

    def test_precision_metrics_update_once(self):
        """
        Test - Updating Precision metrics once return correct value.
        """
        precision = Precision(decimal_places=10)

        true = np.array([1, 0, 0, 1, 0, 1, 1, 0, 0, 1])
        pred = np.array([0.8, 1, 0, 0, 0.6, 0, 1, 0.9, 0, 1])

        expected = 3 / (3 + 3)

        precision.update(true=true, pred=pred)
        actual = precision.get_metric()

        self.assertAlmostEqual(expected, actual)

    def test_precision_metrics_update_several_times(self):
        """
        Test - Updating Precision metrics several times returns
        correct value.
        """
        precision = Precision(decimal_places=10)

        true1 = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1])
        pred1 = np.array([0.8, 0.2, 0.1, 0.9, 0.7, 1, 0.3, 1, 0.1, 1])
        tp1 = 4
        tp_fp1 = tp1 + 2

        true2 = np.array([1, 0, 0, 1, 0, 1, 0])
        pred2 = np.array([0.8, 1, 0.4, 0.45, 1, 0.7, 0.3])
        tp2 = 2
        tp_fp2 = tp2 + 2

        true3 = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        pred3 = np.array([0.7, 0.2, 0.9, 0.5, 0.2, 1, 1, 0])
        tp3 = 3
        tp_fp3 = tp3 + 2

        tp = tp1 + tp2 + tp3
        tp_fp = tp_fp1 + tp_fp2 + tp_fp3
        expected = tp / tp_fp

        precision.update(true=true1, pred=pred1)
        precision.update(true=true2, pred=pred2)
        precision.update(true=true3, pred=pred3)
        actual = precision.get_metric()

        self.assertAlmostEqual(expected, actual)

    def test_precision_metrics_reset_after_update(self):
        """
        Test - Resetting Precision after update returns 0
        """
        precision = Precision(decimal_places=10)

        true = np.array([1, 0, 0, 1, 0, 1, 1, 0, 0, 1])
        pred = np.array([1, 1, 0, 0, 1, 0, 1, 1, 0, 1])

        precision.update(true=true, pred=pred)
        precision.reset()

        self.assertEqual(0, precision.get_metric())

    def test_precision_metrics_no_updates(self):
        """
        Test - Precision metric returns 0 if there hasn't
        been any updates.
        """
        precision = Precision(decimal_places=10)
        self.assertEqual(0, precision.get_metric())

    def test_recall_metrics_correct_initialization(self):
        """
        Test - Recall metrics initialization works with correct
        parameters
        """
        for i in range(11):
            Recall(decimal_places=i, threshold=0.8)

    def test_recall_metrics_decimal_places_incorrect_type_on_init(self):
        """
        Test - Initializing Recall with decimal places of wrong type
        raises a TypeError.
        """
        with self.assertRaises(TypeError):
            Recall(decimal_places=2.5)
        with self.assertRaises(TypeError):
            Recall(decimal_places=[1])
        with self.assertRaises(TypeError):
            Recall("1")

    def test_recall_metrics_decimal_places_incorrect_value_on_init(self):
        """
        Test - Initializing Recall with decimal places out of range
        raises ValueError.
        """
        with self.assertRaises(ValueError):
            Recall(decimal_places=-1)
        with self.assertRaises(ValueError):
            Recall(decimal_places=11)

    def test_recall_metrics_threshold_incorrect_type_on_init(self):
        """
        Test - Initializing Recall with threshold of wrong type
        raises TypeError.
        """
        with self.assertRaises(TypeError):
            Recall(threshold=7)
        with self.assertRaises(TypeError):
            Recall(threshold="0.5")
        with self.assertRaises(TypeError):
            Recall(threshold=[0.5])

    def test_recall_metrics_counts_thresholds_correctly(self):
        """
        Test - Updating Recall with specific threshold works
        correctly.
        """
        recall = Recall(decimal_places=10,
                        threshold=0.8)
        true = np.array([1, 0, 0, 1, 1, 0, 1, 0])
        pred = np.array([0.9, 0.8, 0.7, 0.8, 0.1, 0.6, 1, 0.8])

        expected = 3 / (3 + 1)

        recall.update(true=true, pred=pred)
        actual = recall.get_metric()

        self.assertAlmostEqual(expected, actual)

    def test_recall_metrics_update_once(self):
        """
        Test - Updating Recall metrics once return correct value.
        """
        recall = Recall(decimal_places=10)

        true = np.array([1, 0, 0, 1, 0, 1, 1, 0, 0, 1])
        pred = np.array([0.8, 1, 0, 0, 0.6, 0, 1, 0.9, 0, 1])

        expected = 3 / (3 + 2)

        recall.update(true=true, pred=pred)
        actual = recall.get_metric()

        self.assertAlmostEqual(expected, actual)

    def test_recall_metrics_update_several_times(self):
        """
        Test - Updating Recall metrics several times returns
        correct value.
        """
        recall = Recall(decimal_places=10)

        true1 = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1])
        pred1 = np.array([0.8, 0.2, 0.1, 0.9, 0.7, 1, 0.3, 1, 0.1, 1])
        tp1 = 4
        tp_fn1 = tp1 + 2

        true2 = np.array([1, 0, 0, 1, 0, 1, 0])
        pred2 = np.array([0.8, 1, 0.4, 0.45, 1, 0.7, 0.3])
        tp2 = 2
        tp_fn2 = tp2 + 1

        true3 = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        pred3 = np.array([0.7, 0.2, 0.9, 0.5, 0.2, 1, 1, 0])
        tp3 = 3
        tp_fn3 = tp3 + 1

        tp = tp1 + tp2 + tp3
        tp_fn = tp_fn1 + tp_fn2 + tp_fn3
        expected = tp / tp_fn

        recall.update(true=true1, pred=pred1)
        recall.update(true=true2, pred=pred2)
        recall.update(true=true3, pred=pred3)
        actual = recall.get_metric()

        self.assertAlmostEqual(expected, actual)

    def test_recall_metrics_reset_after_update(self):
        """
        Test - Resetting Recall after update returns 0
        """
        recall = Recall(decimal_places=10)

        true = np.array([1, 0, 0, 1, 0, 1, 1, 0, 0, 1])
        pred = np.array([1, 1, 0, 0, 1, 0, 1, 1, 0, 1])

        recall.update(true=true, pred=pred)
        recall.reset()

        self.assertEqual(0, recall.get_metric())

    def test_recall_metrics_no_updates(self):
        """
        Test - Recall metric returns 0 if there hasn't
        been any updates.
        """
        recall = Recall(decimal_places=10)
        self.assertEqual(0, recall.get_metric())


if __name__ == "__main__":
    unittest.main(verbosity=2)
