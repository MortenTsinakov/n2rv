"""Recall metric file."""

import numpy as np
from n2rv.metrics.metrics import Metrics


class Recall(Metrics):
    """Recall metrics class."""

    def __init__(self,
                 decimal_places: int = 5,
                 threshold: float = 0.5) -> None:
        """
        Recall initializer.

        Inputs:
            decimal_places (int) - the number of decimal places to round
                the result to.
            threshold (float) - predictions >= to this value are considered as
                true (1) and below this value as false (0)
        """
        super().__init__("Recall")
        self.validate_inputs(decimal_places, threshold)
        self.true_positives = 0
        self.false_negatives = 0
        self.decimal_places = decimal_places
        self.threshold = threshold

    def update(self, true: np.ndarray, pred: np.ndarray) -> None:
        """
        Update the metric.

        Inputs:
            true (np.ndarray) - true labels
            pred (np.ndarray) - prediction from the model
        """
        self.true_positives += np.sum((true == 1) & (pred >= self.threshold))
        self.false_negatives += np.sum((true == 1) & (pred < self.threshold))

    def get_metric(self) -> float:
        """
        Return the recall metric.

        Calculates the recall by dividing the true positives by the
        sum of true positives and false negatives.
        """
        if self.true_positives + self.false_negatives > 0:
            denominator = self.true_positives + self.false_negatives
            recall = self.true_positives / denominator
            return round(recall, self.decimal_places)
        return 0

    def reset(self) -> None:
        """Reset the recall metric variables."""
        self.true_positives = 0
        self.false_negatives = 0

    def validate_inputs(self, decimal_places, threshold) -> None:
        self.validate_decimal_places(decimal_places)
        self.validate_threshold(threshold)

    def validate_decimal_places(self, decimal_places) -> None:
        """
        Validate decimal places on initialization

        Inputs:
            decimal_places (int) - The number of decimal places to
                round the result to.
        """
        if not isinstance(decimal_places, int):
            raise TypeError(
                "The argument decimal_places in Recall metric"
                + "should be of type int."
                + f"Got {type(decimal_places)}"
            )
        if decimal_places < 0:
            raise ValueError(
                "The number of decimal places in Recall"
                + f"metric should be >= 0. Got {decimal_places}"
            )
        if decimal_places > 10:
            raise ValueError(
                "The number of decimal places in Recall"
                + f"metric should be <= 10. Got {decimal_places}"
            )

    def validate_threshold(self, threshold) -> None:
        """
        Validate threshold parameter on initialization.

        Inputs:
            threshold (float) - The threshold from where upward the
                predictions are counted as true(1).
        """
        if not isinstance(threshold, float):
            raise TypeError(
                "The threshold paremeter in Recall metric should be " +
                f"of type float. Got: {type(threshold)}."
            )
