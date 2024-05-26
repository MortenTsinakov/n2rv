"""Precision metric file."""

import numpy as np
from n2rv.metrics.metrics import Metrics


class Precision(Metrics):
    """Precision metric class."""

    def __init__(self,
                 decimal_places: int = 5,
                 threshold: float = 0.5) -> None:
        """
        Precision initializer.

        Inputs:
            decimal_places (int) - the number of decimal places to round
                the result to
            threshold (float) - predictions >= to this value are considered as
                true (1) and below this value as false (0)
        """
        super().__init__("Precision")
        self.validate_inputs(decimal_places=decimal_places,
                             threshold=threshold)
        self.true_positives = 0
        self.false_positives = 0
        self.decimal_places = decimal_places
        self.threshold = threshold

    def update(self, true: np.ndarray, pred: np.ndarray) -> None:
        """
        Update the metric.

        Inputs:
            true (np.ndarray) - true labels
            pred (np.ndarray) - predictions from the model
        """
        self.true_positives += np.sum((true == 1) & (pred >= self.threshold))
        self.false_positives += np.sum((true == 0) & (pred >= self.threshold))

    def get_metric(self) -> float:
        """
        Return the precision metric.

        Calculates the precision by dividing the amount of true positives
        with the sum of true positives and false positives.
        """
        if self.true_positives + self.false_positives > 0:
            denominator = self.true_positives + self.false_positives
            precision = self.true_positives / denominator
            return round(precision, self.decimal_places)
        return 0

    def reset(self):
        """Reset the metrics variables."""
        self.true_positives = 0
        self.false_positives = 0

    def validate_inputs(self, decimal_places, threshold):
        """
        Validate inputs on initialization.

        Inputs:
            decimal_places (int) - the number of decimal places that
                should be displayed.
        """
        self.validate_decimal_places(decimal_places=decimal_places)
        self.validate_threshold(threshold=threshold)

    def validate_decimal_places(self, decimal_places: int) -> None:
        """
        Validate decimal places in initialization.

        Inputs:
            decimal_places (int) - The number of decimal places that should
                be displayed
        """
        if not isinstance(decimal_places, int):
            raise TypeError(
                "The argument decimal_places in Precision metric"
                + "should be of type int."
                + f"Got {type(decimal_places)}"
            )
        if decimal_places < 0:
            raise ValueError(
                "The number of decimal places in Precision"
                + f"metric should be >= 0. Got {decimal_places}"
            )
        if decimal_places > 10:
            raise ValueError(
                "The number of decimal places in Precision"
                + f"metric should be <= 10. Got {decimal_places}"
            )

    def validate_threshold(self, threshold: float) -> None:
        """
        Validate threshold parameter on initialization.

        Inputs:
            threshold (float) - The threshold from where upward the
                predictions are counted as true(1).
        """
        if not isinstance(threshold, float):
            raise TypeError(
                "The threshold paremeter in Precision metric should be " +
                f"of type float. Got: {type(threshold)}."
            )
