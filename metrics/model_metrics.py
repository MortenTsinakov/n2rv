"""The model metrics file"""


import numpy as np
from metrics.metrics import Metrics
from exceptions.exception import DuplicateItemsError


class ModelMetrics:
    """
    The model metrics class.
    
    Takes care of updating, printing and reseting all metrics that were defined
    on model compilation.
    """

    def __init__(self, metrics_list: list[Metrics]) -> None:
        self.validate_inputs(metrics_list)
        self.metrics_list = [] if metrics_list is None else metrics_list

    def update_metrics(self, true: np.ndarray, pred: np.ndarray) -> None:
        """Update all metrics."""
        for metric in self.metrics_list:
            metric.update(true, pred)

    def get_metrics(self, loss: float) -> None:
        """Get text to print for all metrics."""
        text = f"Loss: {round(loss, 7)}"
        for metric in self.metrics_list:
            text += f", {metric.get_name()}: {metric.get_metric()}"
        return text

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        for metric in self.metrics_list:
            metric.reset()

    def validate_inputs(self, metrics_list: list[Metrics]) -> None:
        """Make sure that all the metrics arguments are of type Metrics."""
        if metrics_list is None:
            return
        if not isinstance(metrics_list, list):
            raise TypeError(f"Metrics argument on model compilation should be of type list. Got: {type(metrics_list)}.")
        for metric in metrics_list:
            if not isinstance(metric, Metrics):
                raise TypeError(f"Each metric in metrics list should be of type Metrics. Got {type(metric)}")
        if len(metrics_list) > len(set(metrics_list)):
            raise DuplicateItemsError("Metrics list contains duplicates of the same metric.")
