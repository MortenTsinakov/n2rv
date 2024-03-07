"""The parent metrics file."""

import numpy as np


class Metrics:
    """The parent metrics class."""

    def __init__(self, name) -> None:
        self.name = name

    def update(self, true: np.ndarray, pred: np.ndarray) -> None:
        """
        Update the metrics.

        Inputs:
            true (np.ndarray) - true labels
            pred (np.ndarray) - predictions from the model
        """
        raise NotImplementedError

    def get_metric(self) -> float:
        """
        Return the final metric.

        Return:
            float - the final calculated metric value.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset metrics after epoch."""
        raise NotImplementedError

    def get_name(self) -> str:
        """
        Return:
            str - the name of the metric.
        """
        return self.name

    def __eq__(self, other) -> bool:
        """Compare two Metric instances by their names."""
        if isinstance(other, type(self)):
            return self.name == other.name
        return False

    def __hash__(self) -> int:
        return hash(self.name)
