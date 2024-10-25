from abc import ABC, abstractmethod
from typing import Any
import numpy as np

METRICS = [
    "mean_squared_error",
    "accuracy",
]  # add the names (in strings) of the metrics you implement


def get_metric(name: str):
    # Factory function to get a metric by name.
    # Return a metric instance given its str name.

    if name == "mean_squared_error":
        return MeanSquaredError()
    elif name == "accuracy":
        return Accuracy()


class Metric(ABC):
    """Base class for all metrics.
    """
    # your code here
    # remember: metrics take ground truth and prediction as input and return a real number

    def __call__(self):
        return self.evaluate()

    @abstractmethod
    def evaluate(self):
        pass


# add here concrete implementations of the Metric class
# Metric implementations for regression
class MeanSquaredError(Metric):
    def __init__(self, y_hat=None, y=None):
        self.y_hat = y_hat
        self.y = y

    def evaluate(self, y_hat, y):
        return np.mean((y_hat - y) ** 2)


class MeanAbsoluteError(Metric):
    def __init__(self, y_hat=None, y=None):
        self.y_hat = y_hat
        self.y = y

    def evaluate(self, y_hat, y):
        return np.mean(np.abs(y_hat - y))


class RSquared(Metric):
    def __init__(self, y_hat=None, y=None):
        self.y_hat = y_hat
        self.y = y

    def evaluate(self, y_hat, y):
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


# Metric implementations for classification
class Accuracy(Metric):
    def __init__(self, y_hat=None, y=None):
        self.y_hat = y_hat
        self.y = y

    def evaluate(self, y_hat, y):
        return np.mean(y_hat == y)


class Precision(Metric):
    def __init__(self, y_hat=None, y=None):
        self.y_hat = y_hat
        self.y = y

    def evaluate(self, y_hat, y):
        true_positive = np.sum((y_hat == 1) & (y == 1))
        false_positive = np.sum((y_hat == 1) & (y == 0))
        return true_positive / (true_positive + false_positive)


class Recall(Metric):
    def __init__(self, y_hat=None, y=None):
        self.y_hat = y_hat
        self.y = y

    def evaluate(self, y_hat, y):
        true_positive = np.sum((y_hat == 1) & (y == 1))
        false_negative = np.sum((y_hat == 0) & (y == 1))
        return true_positive / (true_positive + false_negative)