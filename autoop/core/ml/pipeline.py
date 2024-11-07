"""Pipeline class for managing machine learning workflows."""

from typing import List
import pickle

from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model import Model
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.functional.preprocessing import preprocess_features
import numpy as np


class Pipeline():
    """Pipeline class for managing machine learning workflows."""

    def __init__(self: "Pipeline",
                 metrics: List[Metric],
                 dataset: Dataset,
                 model: Model,
                 input_features: List[Feature],
                 target_feature: Feature,
                 split: float = 0.8,
                 ) -> None:
        """Initialize a new pipeline.

        Args:
            metrics (List[Metric]): List of metrics.
            dataset (Dataset): Dataset object.
            model (Model): Model object.
            input_features (List[Feature]): List of input features.
            target_feature (Feature): Target feature.
            split (float, optional): Split ratio for training and testing data.
        """
        self._dataset = dataset
        self._model = model
        self._input_features = input_features
        self._target_feature = target_feature
        self._metrics = metrics
        self._artifacts = {}
        self._split = split
        if (target_feature.type == "categorical" and
                model.type != "classification"):
            raise ValueError(
                "Model type must be classification for "
                "categorical target feature"
            )
        if target_feature.type == "continuous" and model.type != "regression":
            raise ValueError(
                "Model type must be regression for continuous target feature"
            )

    def __str__(self: "Pipeline") -> str:
        """Return the string representation of the pipeline.

        returns:
            str: String representation of the pipeline.
        """
        return f"""
        Pipeline(
        model={self._model.type},
        input_features={list(map(str, self._input_features))},
        target_feature={str(self._target_feature)},
        split={self._split},
        metrics={list(map(str, self._metrics))},
        )
        """

    @property
    def model(self: "Pipeline") -> Model:
        """Used to get the model object."""
        return self._model

    @property
    def artifacts(self: "Pipeline") -> List[Artifact]:
        """Used to get the artifact generated during the pipeline execution."""
        artifacts = []
        for name, artifact in self._artifacts.items():
            artifact_type = artifact.get("type")
            if artifact_type in ["OneHotEncoder"]:
                data = artifact["encoder"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
            if artifact_type in ["StandardScaler"]:
                data = artifact["scaler"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
        pipeline_data = {
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
        }
        artifacts.append(
            Artifact(
                name="pipeline_config",
                data=pickle.dumps(pipeline_data)
            )
        )
        artifacts.append(
            self._model.to_artifact(name=f"pipeline_model_{self._model.type}")
        )
        return artifacts

    def _register_artifact(
        self: "Pipeline", name: str, artifact: Artifact
    ) -> None:
        """Register an artifact generated during the pipeline execution.

        Args:
            name (str): Name of the artifact.
            artifact (Artifact): Artifact object.
        """
        self._artifacts[name] = artifact

    def _preprocess_features(self: "Pipeline") -> None:
        """Preprocess the features."""
        (target_feature_name, target_data, artifact) = preprocess_features(
            [self._target_feature], self._dataset
        )[0]
        self._register_artifact(target_feature_name, artifact)
        input_results = preprocess_features(
            self._input_features, self._dataset
        )
        for (feature_name, data, artifact) in input_results:
            self._register_artifact(feature_name, artifact)
        # Get the input vectors and output vector,
        # sort by feature name for consistency
        self._output_vector = target_data
        self._input_vectors = [
            data for (feature_name, data, artifact) in input_results
        ]

    def _split_data(self: "Pipeline") -> None:
        """Split the data into training and testing sets."""
        split = self._split
        self._train_X = [
            vector[:int(split * len(vector))]
            for vector in self._input_vectors
        ]
        self._test_X = [
            vector[int(split * len(vector)):]
            for vector in self._input_vectors
        ]
        self._train_y = self._output_vector[
            :int(split * len(self._output_vector))
        ]
        self._test_y = self._output_vector[
            int(split * len(self._output_vector)):]

    def _compact_vectors(
        self: "Pipeline", vectors: List[np.array]
    ) -> np.array:
        return np.concatenate(vectors, axis=1)

    def _train(self: "Pipeline") -> None:
        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._model.fit(X, Y)

    def _evaluate(self: "Pipeline") -> None:
        X = self._compact_vectors(self._test_X)
        Y = self._test_y
        self._metrics_results = []
        predictions = self._model.predict(X)
        for metric in self._metrics:
            result = metric.evaluate(predictions, Y)
            self._metrics_results.append((metric, result))
        self._predictions = predictions

    def execute(self: "Pipeline") -> dict:
        """Execute the pipeline.

        Returns:
            dict: Dictionary containing metrics and predictions.
        """
        self._preprocess_features()
        self._split_data()
        self._train()
        self._evaluate()
        return {
            "metrics": self._metrics_results,
            "predictions": self._predictions,
        }
