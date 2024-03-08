"""Run all tests at once."""

import unittest


def create_test_suite():
    """Discover all test files and return the suite."""
    test_suite = unittest.TestLoader().discover("tests", pattern="test*.py")
    return test_suite


if __name__ == "__main__":
    suite = create_test_suite()
    unittest.TextTestRunner(verbosity=2).run(suite)
