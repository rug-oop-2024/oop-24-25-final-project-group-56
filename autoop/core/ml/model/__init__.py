"""This module provides a factory function to get models by name."""
from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression.multiple_linear_regression import (
    MultipleLinearRegression,
)
from autoop.core.ml.model.regression.lasso_regression import LassoRegression
from autoop.core.ml.model.regression.mlp_regressor import MLPRegressor
from autoop.core.ml.model.classification.k_neighbors_classifier import (
    KNClassifier,
)
from autoop.core.ml.model.classification.mlp_classifier import MLPClassifier
from autoop.core.ml.model.classification.ridge_classifier import (
    RidgeClassifier,
)

REGRESSION_MODELS = [
    "MultipleLinearRegression",
    "LassoRegression",
    "MLPRegressor",
]  # add your models as str here

CLASSIFICATION_MODELS = [
    "KNClassifier",
    "MLPClassifier",
    "RidgeClassifier",
]  # add your models as str here


def get_model(model_name: str) -> Model:
    """Get a model by name."""
    if model_name in REGRESSION_MODELS:
        if model_name == "MultipleLinearRegression":
            return MultipleLinearRegression()
        elif model_name == "LassoRegression":
            return LassoRegression()
        elif model_name == "MLPRegressor":
            return MLPRegressor()
    if model_name in CLASSIFICATION_MODELS:
        if model_name == "MLPClassifier":
            return MLPClassifier()
        elif model_name == "RidgeClassifier":
            return RidgeClassifier()
        elif model_name == "KNClassifier":
            return KNClassifier()
