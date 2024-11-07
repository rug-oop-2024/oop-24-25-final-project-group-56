"""This module contains the implementation of various metrics."""
from abc import ABC, abstractmethod
import numpy as np

METRICS = [
    "mean_squared_error",
    "accuracy",
    "mean_absolute_error",
    "r_squared",
    "precision",
    "recall",
]  # add the names (in strings) of the metrics you implement


def get_metric(name: str) -> "Metric":
    """Get a metric by name."""
    if name == "mean_squared_error" or name == "MeanSquaredError":
        return MeanSquaredError()
    elif name == "accuracy" or name == "Accuracy":
        return Accuracy()
    elif name == "mean_absolute_error" or name == "MeanAbsoluteError":
        return MeanAbsoluteError()
    elif name == "r_squared" or name == "RSquared":
        return RSquared()
    elif name == "precision" or name == "Precision":
        return Precision()
    elif name == "recall" or name == "Recall":
        return Recall()


class Metric(ABC):
    """Base class for all metrics."""

    # your code here
    # remember: metrics take ground truth and prediction as input
    # and return a real number

    def __call__(self: "Metric") -> float:
        """Call the metric."""
        return self.evaluate()

    def __str__(self: "Metric") -> str:
        """Return the string representation of the metric."""
        return self.__class__.__name__

    @abstractmethod
    def evaluate(self: "Metric") -> None:
        """Evaluate the metric."""
        pass


# add here concrete implementations of the Metric class
# Metric implementations for regression
class MeanSquaredError(Metric):
    """Mean Squared Error metric.

    args:
        y_hat: predicted values
        y: true values
    returns:
        mean squared error
    """

    def __init__(
        self: "MeanSquaredError",
        y_hat: np.ndarray = None,
        y: np.ndarray = None
    ) -> None:
        """
        Initialize the MeanSquaredError class.

        y_hat: predicted values
        y: true values
        """
        self.y_hat = y_hat
        self.y = y

    def evaluate(
        self: "MeanSquaredError",
        y_hat: np.ndarray,
        y: np.ndarray
    ) -> float:
        """
        Evaluate the mean squared error.

        args:
        y_hat: predicted values
        y: true values
        returns:
            mean squared error
        """
        return np.mean((y_hat - y) ** 2)


class MeanAbsoluteError(Metric):
    """
    Mean Absolute Error metric.

    args:
        y_hat: predicted values
        y: true values
    returns:
        mean absolute error
    """

    def __init__(
        self: "MeanAbsoluteError",
        y_hat: np.ndarray = None,
        y: np.ndarray = None
    ) -> None:
        """
        Initialize the MeanAbsoluteError class.

        args:
        y_hat: predicted values
        y: true values
        """
        self.y_hat = y_hat
        self.y = y

    def evaluate(
        self: "MeanAbsoluteError",
        y_hat: np.ndarray,
        y: np.ndarray
    ) -> float:
        """
        Evaluate the mean absolute error.

        args:
        y_hat: predicted values
        y: true values
        returns:
            mean absolute error
        """
        return np.mean(np.abs(y_hat - y))


class RSquared(Metric):
    """
    R Squared metric.

    args:
        y_hat: predicted values
        y: true values
    returns:
        r squared value
    """

    def __init__(
        self: "RSquared",
        y_hat: np.ndarray = None,
        y: np.ndarray = None
    ) -> None:
        """
        Initialize the RSquared class.

        args:
        y_hat: predicted values
        y: true values
        """
        self.y_hat = y_hat
        self.y = y

    def evaluate(self: "RSquared", y_hat: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate the r squared value.

        args:
        y_hat: predicted values
        y: true values
        returns:
            r squared value
        """
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


# Metric implementations for classification
class Accuracy(Metric):
    """
    Accuracy metric.

    args:
        y_hat: predicted values
        y: true values
    returns:
        accuracy
    """

    def __init__(
        self: "Accuracy",
        y_hat: np.ndarray = None,
        y: np.ndarray = None
    ) -> None:
        """
        Initialize the Accuracy class.

        args:
        y_hat: predicted values
        y: true values
        """
        self.y_hat = y_hat
        self.y = y

    def evaluate(self: "Accuracy", y_hat: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate the accuracy.

        args:
        y_hat: predicted values
        y: true values
        returns:
            accuracy
        """
        return np.mean(y_hat == y)


class Precision(Metric):
    """
    Precision metric.

    args:
        y_hat: predicted values
        y: true values
    returns:
        precision
    """

    def __init__(
        self: "Precision",
        y_hat: np.ndarray = None,
        y: np.ndarray = None
    ) -> None:
        """
        Initialize the Precision class.

        args:
        y_hat: predicted values
        y: true values
        """
        self.y_hat = y_hat
        self.y = y

    def evaluate(self: "Precision", y_hat: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate the precision.

        args:
        y_hat: predicted values
        y: true values
        returns:
            precision
        """
        true_positive = np.sum((y_hat == 1) & (y == 1))
        false_positive = np.sum((y_hat == 1) & (y == 0))
        return true_positive / (true_positive + false_positive)


class Recall(Metric):
    """
    Recall metric.

    args:
        y_hat: predicted values
        y: true values
    returns:
        recall
    """

    def __init__(
        self: "Recall",
        y_hat: np.ndarray = None,
        y: np.ndarray = None
    ) -> None:
        """
        Initialize the Recall class.

        args:
        y_hat: predicted values
        y: true values
        """
        self.y_hat = y_hat
        self.y = y

    def evaluate(self: "Recall", y_hat: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate the recall.

        args:
        y_hat: predicted values
        y: true values
        returns:
            recall
        """
        true_positive = np.sum((y_hat == 1) & (y == 1))
        false_negative = np.sum((y_hat == 0) & (y == 1))
        return true_positive / (true_positive + false_negative)
