"""Accuracy metric file."""


import numpy as np
from metrics.metrics import Metrics


class Accuracy(Metrics):
    """Accuracy metric class."""

    def __init__(self, decimal_places: int = 5) -> None:
        """
        Accuracy initializer.
        
        Inputs:
            decimal_places (int) - The number of decimal places to round the result to.
        """
        super().__init__("Accuracy")
        self.validate_inputs(decimal_places)
        self.correct = 0
        self.total = 0
        self.decimal_places = decimal_places

    def update(self, true: np.ndarray, pred: np.ndarray) -> None:
        """
        Update the metric.
        
        Inputs:
            true (np.ndarray) - true labels
            pred (np.ndarray) - predictions from the model
        """
        self.correct += np.sum(np.argmax(true, axis=1) == np.argmax(pred, axis=1))
        self.total += len(true)

    def get_metric(self) -> float:
        """
        Return the accuracy metric.

        Calculates the accuracy by dividing the correct predictions by total predictions.
        """
        if self.total:
            return round(self.correct / self.total, self.decimal_places)
        return 0

    def reset(self) -> None:
        """Reset the metrics after an epoch."""
        self.correct = 0
        self.total = 0

    def validate_inputs(self, decimal_places) -> None:
        """Validate inputs on initialization."""
        if not isinstance(decimal_places, int):
            raise TypeError(f"The argument decimal_places in Accuracy metric should be of type int. Got {type(decimal_places)}")
        if decimal_places < 0:
            raise ValueError(f"The number of decimal places in Accuracy metric should be >= 0. Got {decimal_places}")
        if decimal_places > 10:
            raise ValueError(f"The number of decimal places in Accuracy metric should be <= 10. Got {decimal_places}")

    def get_name(self) -> str:
        """Return the name of the metric."""
        return self.name
