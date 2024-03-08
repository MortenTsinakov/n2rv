"""The model metrics file"""

import numpy as np
from n2rv.exceptions.exception import DuplicateItemsError
from n2rv.metrics.metrics import Metrics


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
        """
        Update all metrics.

        Inputs:
            true (np.ndarray) - the true values
            pred (np.ndarray) - the predicted values
        """
        for metric in self.metrics_list:
            metric.update(true, pred)

    def get_metrics(self, loss: float) -> dict[str, Metrics]:
        """
        Get dictionary with metric names as keys and calculated
        errors as values.

        Inputs.
            loss (float) - the loss calculated by the Model class.
        Return:
            dict - dictionary of all requested metrics in the form
                {<metric name> : <calculated error>}
        """
        metrics_dict = {"Loss": round(loss, 7)}
        for metric in self.metrics_list:
            metrics_dict[metric.get_name()] = metric.get_metric()
        return metrics_dict

    def reset_metrics(self) -> None:
        """Reset metric variables for all metrics."""
        for metric in self.metrics_list:
            metric.reset()

    def validate_inputs(self, metrics_list: list[Metrics]) -> None:
        """Make sure that all the metrics arguments are of type Metrics."""
        if metrics_list is None:
            return
        if not isinstance(metrics_list, list):
            raise TypeError(
                "Metrics argument on model compilation should"
                + f"be of type list. Got: {type(metrics_list)}."
            )
        for metric in metrics_list:
            if not isinstance(metric, Metrics):
                raise TypeError(
                    "Each metric in metrics list should be of"
                    + f"type Metrics. Got {type(metric)}"
                )
        if len(metrics_list) > len(set(metrics_list)):
            raise DuplicateItemsError(
                "Metrics list contains duplicates of" + "the same metric."
            )
