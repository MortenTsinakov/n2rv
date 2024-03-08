"""Binary accuracy metric file."""


import numpy as np
from n2rv.metrics.metrics import Metrics


class BinaryAccuracy(Metrics):
    """Binary Accuracy metric class."""

    def __init__(self,
                 decimal_places: int = 5,
                 threshold: float = 0.5) -> None:
        """
        Binary Accuracy initializer.
        
        Inputs:
            decimal_places (int) - the number of decimal places to round the
                result to
            threshold (float) - predictions over this value are considered as 1
                and below this value as 0
        """
        super().__init__("Binary Accuracy")
        self.validate_inputs(decimal_places, threshold)
        self.correct = 0
        self.total = 0
        self.decimal_places = decimal_places
        self.threshold = threshold

    def update(self, true: np.ndarray, pred: np.ndarray) -> None:
        """
        Update the metric.

        Inputs:
            true (np.ndarray) - true labels
            pred (np.ndarray) - predicted labels
        """
        self.correct += np.count_nonzero((pred >= self.threshold) == true)
        self.total += len(true)

    def get_metric(self) -> float:
        """
        Return the accuracy metric

        Calculate the accuracy by dividing the correct predictions by total
        number of predictions.

        Return:
            float: accuracy of predictions in range [0, 1]
        """
        if self.total:
            return round(self.correct / self.total, self.decimal_places)
        return 0

    def reset(self) -> None:
        """Reset the metrics variables."""
        self.correct = 0
        self.total = 0

    def validate_inputs(self, decimal_places: int, threshold: float) -> None:
        self.validate_decimal_places(decimal_places)
        self.validate_threshold(threshold)

    def validate_decimal_places(self, decimal_places: int) -> None:
        if not isinstance(decimal_places, int):
            raise TypeError(
                "The argument decimal_places in Binary Accuracy metric "
                + "should be of type int. "
                + f"Got {type(decimal_places)}"
            )
        if decimal_places < 0:
            raise ValueError(
                "The number of decimal places in Binary Accuracy "
                + f"metric should be >= 0. Got {decimal_places}"
            )
        if decimal_places > 10:
            raise ValueError(
                "The number of decimal places in Binary Accuracy "
                + f"metric should be <= 10. Got {decimal_places}"
            )

    def validate_threshold(self, threshold: float) -> None:
        if not isinstance(threshold, float):
            raise TypeError("The threshold argument in Binary Accuracy " +
                            "metric should be of type float. " +
                            f"Got {type(threshold)}")
