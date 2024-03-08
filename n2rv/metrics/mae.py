"""The file for Mean Absolute Error metric."""

import numpy as np
from n2rv.metrics.metrics import Metrics


class MAE(Metrics):
    """Class for Mean Absolute Error metric."""

    def __init__(self, decimal_places: int = 5) -> None:
        super().__init__("Mean Absolute Error")
        self.validate_inputs(decimal_places)
        self.error = 0
        self.n_samples = 0
        self.decimal_places = decimal_places

    def update(self, true: np.ndarray, pred: np.ndarray) -> None:
        """
        Update the metric.

        Inputs:
            true (np.ndarray) - true labels
            pred (np.ndarray) - predicted labels
        """
        self.error += np.sum(np.absolute(true - pred))
        self.n_samples += len(true)

    def get_metric(self) -> float:
        """
        Return the Mean Absolute Error metric.

        Calculates the MAE metric by dividing summed abosolute error over all
        samples by the number of samples.

        Return:
            float - mean average error between the actual values and the
                predictions.
        """
        if self.n_samples:
            return round(self.error / self.n_samples, self.decimal_places)
        return 0

    def reset(self) -> None:
        """Reset the metric variables."""
        self.error = 0
        self.n_samples = 0

    def validate_inputs(self, decimal_places) -> None:
        """Validate inputs on initialization."""
        if not isinstance(decimal_places, int):
            raise TypeError(
                "The argument decimal_places in MAE metric"
                + "should be of type int."
                + f"Got {type(decimal_places)}"
            )
        if decimal_places < 0:
            raise ValueError(
                "The number of decimal places in MAE metric"
                + f"should be >= 0. Got {decimal_places}"
            )
        if decimal_places > 10:
            raise ValueError(
                "The number of decimal places in MAE metric"
                + f"should be <= 10. Got {decimal_places}"
            )
