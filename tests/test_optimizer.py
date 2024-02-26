"""Tests for optimizers."""


import unittest
import os
import sys


# Get the absolute path of the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Paths to files in the directories above
paths = [
    os.path.join(current_script_dir, '..', 'optimizers')
]

# Add all paths
sys.path += [os.path.dirname(name) for name in paths]

# Import files for testing
from optimizers.adam import Adam
from optimizers.momentum import Momentum
from optimizers.rmsprop import RMSProp
from optimizers.sgd import SGD


class TestOptimizers(unittest.TestCase):
    def test_sgd_lr_not_float(self):
        """Test - Initializing SGD with non-float learning rate fails."""
        fail_text = "Exception should be raised on SGD initialization if learning rate is not float."
        try:
            SGD(learning_rate=8)
            self.fail(fail_text)
        except TypeError:
            pass

        try:
            SGD([0.5])
            self.fail(fail_text)
        except TypeError:
            pass

        try:
            SGD("0.1")
            self.fail(fail_text)
        except TypeError:
            pass

    def test_sgd_learning_rate_less_than_0(self):
        """Test - Initializing SGD with learning rate value less than 0 fails."""
        fail_text = "Exception should be raised on SGD initialization when learning rate is < 0."
        try:
            SGD(-0.1)
            self.fail(fail_text)
        except ValueError:
            pass

    def test_sgd_learning_rate_is_0(self):
        """Test - Initializing SGD with learning rate of zero fails."""
        fail_text = "Exceptions should be raised on SGD initialization if learning rate is 0."
        try:
            SGD(0.0)
            self.fail(fail_text)
        except ValueError:
            pass

    def test_momentum_learning_rate_not_float(self):
        """Test - Momentum initialization fails if learning rate is not float."""
        fail_text = "Exception should be thrown on Momentum initialization if learning rate is not float."
        
        try:
            Momentum(learning_rate=10)
            self.fail(fail_text)
        except TypeError:
            pass

        try:
            Momentum(learning_rate=[0.1])
            self.fail(fail_text)
        except TypeError:
            pass

        try:
            Momentum("0.1")
            self.fail(fail_text)
        except TypeError:
            pass

    def test_momentum_learning_rate_less_than_0(self):
        """Test - Momentum initialization fails if learning rate is < 0."""
        fail_text = "Exception should be raised on Momentum initialization if learning rate < 0."
        try:
            Momentum(-0.01)
            self.fail(fail_text)
        except ValueError:
            pass

    def test_momentum_learning_rate_is_zero(self):
        """Test - Momentum initialization fails if learning rate is 0."""
        fail_text = "Exception should be raised on Momentum initialization if learning rate is 0."
        try:
            Momentum(learning_rate=0.0)
            self.fail(fail_text)
        except ValueError:
            pass

    def test_momentum_momentum_argument_not_float(self):
        """Test - Momentum initialization fails if momentum argument is not float."""
        fail_text = "Exception should be raised on Momentum initialization if momentum is not float."
        try:
            Momentum(momentum=1)
            self.fail(fail_text)
        except TypeError:
            pass

        try:
            Momentum(momentum=[0.1])
            self.fail(fail_text)
        except TypeError:
            pass

        try:
            Momentum(momentum="0.1")
            self.fail(fail_text)
        except TypeError:
            pass

    def test_momentum_momentum_argument_zero_or_less(self):
        """Test - Momentum initialization fails if momentum argument is <= 0."""
        fail_text = "Exception should be raised on Momentum initialization if momentum <= 0."
        try:
            Momentum(momentum=0.0)
            self.fail(fail_text)
        except ValueError:
            pass

        try:
            Momentum(momentum=-1.2)
            self.fail(fail_text)
        except ValueError:
            pass

    def test_momentum_momentum_argument_1_or_greater(self):
        """Test - Momentum initialization fails if momentum argument >= 1."""
        fail_text = "Exception should be raised on Momentum initialization if momentum >= 1."
        try:
            Momentum(momentum=1.0)
            self.fail(fail_text)
        except ValueError:
            pass

        try:
            Momentum(momentum=1.1)
            self.fail(fail_text)
        except ValueError:
            pass

    def test_rms_prop_learning_rate_is_not_float(self):
        """Test - RMSProp initialization fails if learning rate is not float."""
        fail_text = "Exception should be raised on RMSProp initialization if learning rate is not float."
        try:
            RMSProp(learning_rate=1)
            self.fail(fail_text)
        except TypeError:
            pass

        try:
            RMSProp(learning_rate=[0.1])
            self.fail(fail_text)
        except TypeError:
            pass

        try:
            RMSProp(learning_rate="0.01")
            self.fail(fail_text)
        except TypeError:
            pass

    def test_rms_prop_learning_rate_is_zero_or_less(self):
        """Test - RMSProp initialization fails if learning rate is <= 0."""
        fail_text = "Exception should be raised on RMSProp initialization if learning rate <= 0."
        try:
            RMSProp(learning_rate=0.0)
            self.fail(fail_text)
        except ValueError:
            pass

        try:
            RMSProp(learning_rate=-0.5)
            self.fail(fail_text)
        except ValueError:
            pass

    def test_rms_prop_beta_not_float(self):
        """Test - RMSProp initialization fails if beta is not float."""
        fail_text = "Exception should be raised on RMSProp initialization if beta is not float."
        try:
            RMSProp(beta=1)
            self.fail(fail_text)
        except TypeError:
            pass

        try:
            RMSProp(beta=[0.1])
            self.fail(fail_text)
        except TypeError:
            pass

        try:
            RMSProp(beta="0.1")
            self.fail(fail_text)
        except TypeError:
            pass

    def test_rms_prop_beta_is_zero_or_less(self):
        """Test - RMSProp initialization fails if beta <= 0."""
        fail_text = "Exception should be raised on RMSProp initialization if beta <= 0."
        try:
            RMSProp(beta=0.0)
            self.fail(fail_text)
        except ValueError:
            pass

        try:
            RMSProp(beta=-0.1)
            self.fail(fail_text)
        except ValueError:
            pass

    def test_rms_prop_beta_is_1_or_greater(self):
        """Test - RMSProp initialization fails if beta >= 1."""
        fail_text = "Exception should be raised on RMSProp initialization if beta >= 1."
        try:
            RMSProp(beta=1.0)
            self.fail(fail_text)
        except ValueError:
            pass

        try:
            RMSProp(beta=1.5)
            self.fail(fail_text)
        except ValueError:
            pass

    def test_rms_prop_epsilon_not_float(self):
        """Test - RMSProp initialization fails if epsilon is not float."""
        fail_text = "Exception should be raised on RMSProp initialization if epsilon is not float."
        try:
            RMSProp(epsilon=1)
            self.fail(fail_text)
        except TypeError:
            pass

        try:
            RMSProp(epsilon=[0.1])
            self.fail(fail_text)
        except TypeError:
            pass

        try:
            RMSProp(epsilon="0.1")
            self.fail(fail_text)
        except TypeError:
            pass

    def test_rms_prop_epsilon_is_zero_or_less(self):
        """Test - RMSProp initialization fails if epsilon <= 0."""
        fail_text = "Exception should be raised on RMSProp initialization if epsilon <= 0."
        try:
            RMSProp(epsilon=0.0)
            self.fail(fail_text)
        except ValueError:
            pass

        try:
            RMSProp(epsilon=-0.1)
            self.fail(fail_text)
        except ValueError:
            pass

    def test_adam_learning_rate_is_not_float(self):
        """Test - Adam initialization fails if learning rate is not float."""
        fail_text = "Exception should be raised on Adam initialization if learning rate is not float."
        try:
            Adam(learning_rate=1)
            self.fail(fail_text)
        except TypeError:
            pass

        try:
            Adam(learning_rate=[0.1])
            self.fail(fail_text)
        except TypeError:
            pass


        try:
            Adam(learning_rate="0.2")
            self.fail(fail_text)
        except TypeError:
            pass

    def test_adam_learning_rate_is_zero_or_less(self):
        """Test - Adam initialization fails if learning rate <= 0."""
        fail_text = "Exception should be raised on Adam initialization if learning rate <= 0."
        try:
            Adam(learning_rate=0.0)
            self.fail(fail_text)
        except ValueError:
            pass

        try:
            Adam(learning_rate=-0.5)
            self.fail(fail_text)
        except ValueError:
            pass

    def test_adam_betas_are_not_floats(self):
        """Test - Adam initialization fails if beta1 or beta2 is not a float."""
        fail_text = "Exception should be raised on Adam initialization if beta1 or beta2 is not float."
        try:
            Adam(beta1=1)
            self.fail(fail_text)
        except TypeError:
            pass

        try:
            Adam(beta1=[0.2])
            self.fail(fail_text)
        except TypeError:
            pass

        try:
            Adam(beta1="0.2")
            self.fail(fail_text)
        except TypeError:
            pass

        try:
            Adam(beta2=1)
            self.fail(fail_text)
        except TypeError:
            pass

        try:
            Adam(beta2=[0.2])
            self.fail(fail_text)
        except TypeError:
            pass

        try:
            Adam(beta2="0.2")
            self.fail(fail_text)
        except TypeError:
            pass

    def test_adam_betas_are_zero_or_less(self):
        """Test - Adam initialization fails if beta1 or beta2 <= 0."""
        fail_text = "Exception should be raised on Adam initialization if beta1 <= 0 or beta2 <= 0."
        try:
            Adam(beta1=0.0)
            self.fail(fail_text)
        except ValueError:
            pass

        try:
            Adam(beta1=-0.1)
            self.fail(fail_text)
        except ValueError:
            pass

        try:
            Adam(beta2=0.0)
            self.fail(fail_text)
        except ValueError:
            pass

        try:
            Adam(beta2=-0.1)
            self.fail(fail_text)
        except ValueError:
            pass

    def test_adam_epsilon_is_not_float(self):
        """Test - Adam initialization fails if epsilon is not float."""
        fail_text = "Exception should be raised on Adam initialization if epsilon is not float."
        try:
            Adam(epsilon=1)
            self.fail(fail_text)
        except TypeError:
            pass

        try:
            Adam(epsilon=[0.1])
            self.fail(fail_text)
        except TypeError:
            pass

        try:
            Adam(epsilon="0.1")
            self.fail(fail_text)
        except TypeError:
            pass

    def test_adam_epsilon_is_zero_or_less(self):
        """Test - Adam initialization fails if epsilon <= 0."""
        fail_text = "Exception should be raised on Adam initialization if epsilon <= 0."
        try:
            Adam(epsilon=0.0)
            self.fail(fail_text)
        except ValueError:
            pass

        try:
            Adam(epsilon=-0.1)
            self.fail(fail_text)
        except ValueError:
            pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
