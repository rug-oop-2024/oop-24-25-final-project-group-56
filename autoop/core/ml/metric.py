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


def get_metric(name: str):
    # Factory function to get a metric by name.
    # Return a metric instance given its str name.

    if name == "mean_squared_error":
        return MeanSquaredError()
    elif name == "accuracy":
        return Accuracy()
    elif name == "mean_absolute_error":
        return MeanAbsoluteError()
    elif name == "r_squared":
        return RSquared()
    elif name == "precision":
        return Precision()
    elif name == "recall":
        return Recall()


class Metric(ABC):
    """Base class for all metrics.
    """
    # your code here
    # remember: metrics take ground truth and prediction as input
    # and return a real number

    def __call__(self):
        return self.evaluate()

    def __str__(self):
        return self.__class__.__name__

    @abstractmethod
    def evaluate(self):
        pass


# add here concrete implementations of the Metric class
# Metric implementations for regression
class MeanSquaredError(Metric):
    '''Mean Squared Error metric
    args:
        y_hat: predicted values
        y: true values
    returns:
        mean squared error
    '''
    def __init__(self, y_hat=None, y=None) -> None:
        '''
        y_hat: predicted values
        y: true values
        '''
        self.y_hat = y_hat
        self.y = y

    def evaluate(self, y_hat, y) -> float:
        '''
        y_hat: predicted values
        y: true values
        returns:
            mean squared error
        '''
        return np.mean((y_hat - y) ** 2)


class MeanAbsoluteError(Metric):
    '''Mean Absolute Error metric
    args:
        y_hat: predicted values
        y: true values
    returns:
        mean absolute error
    '''
    def __init__(self, y_hat=None, y=None) -> None:
        '''
        y_hat: predicted values
        y: true values
        '''
        self.y_hat = y_hat
        self.y = y

    def evaluate(self, y_hat, y) -> float:
        '''
        y_hat: predicted values
        y: true values
        returns:
            mean absolute error
        '''
        return np.mean(np.abs(y_hat - y))


class RSquared(Metric):
    '''R Squared metric
    args:
        y_hat: predicted values
        y: true values
    returns:
        r squared value
    '''
    def __init__(self, y_hat=None, y=None) -> None:
        '''
        y_hat: predicted values
        y: true values
        '''
        self.y_hat = y_hat
        self.y = y

    def evaluate(self, y_hat, y) -> float:
        '''
        y_hat: predicted values
        y: true values
        returns:
            r squared value
        '''
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


# Metric implementations for classification
class Accuracy(Metric):
    '''Accuracy metric
    args:
        y_hat: predicted values
        y: true values
    returns:
        accuracy
    '''
    def __init__(self, y_hat=None, y=None):
        '''
        y_hat: predicted values
        y: true values
        '''
        self.y_hat = y_hat
        self.y = y

    def evaluate(self, y_hat, y) -> float:
        '''
        y_hat: predicted values
        y: true values
        returns:
            accuracy
        '''
        return np.mean(y_hat == y)


class Precision(Metric):
    '''Precision metric
    args:
        y_hat: predicted values
        y: true values
    returns:
        precision
    '''
    def __init__(self, y_hat=None, y=None):
        '''
        y_hat: predicted values
        y: true values
        '''
        self.y_hat = y_hat
        self.y = y

    def evaluate(self, y_hat, y) -> float:
        '''
        y_hat: predicted values
        y: true values
        returns:
            precision
        '''
        true_positive = np.sum((y_hat == 1) & (y == 1))
        false_positive = np.sum((y_hat == 1) & (y == 0))
        return true_positive / (true_positive + false_positive)


class Recall(Metric):
    '''Recall metric
    args:
        y_hat: predicted values
        y: true values
    returns:
        recall
    '''
    def __init__(self, y_hat=None, y=None):
        '''
        y_hat: predicted values
        y: true values
        '''
        self.y_hat = y_hat
        self.y = y

    def evaluate(self, y_hat, y) -> float:
        '''
        y_hat: predicted values
        y: true values
        returns:
            recall
        '''
        true_positive = np.sum((y_hat == 1) & (y == 1))
        false_negative = np.sum((y_hat == 0) & (y == 1))
        return true_positive / (true_positive + false_negative)
