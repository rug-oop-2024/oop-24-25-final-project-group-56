"""The user can see existing saved pipelines and deploy them.

The user can the select a pipeline to show the summary of the pipeline.
Once the user loads a pipeline, prompt them to provide a CSV
on which they can perform predictions.
"""
import streamlit as st
import pandas as pd
import io
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.metric import get_metric
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.model import get_model

st.set_page_config(page_title="Deployment", page_icon="🚀")

st.write("# 🚀 Deployment")

automl = AutoMLSystem.get_instance()

pipelines = automl.registry.list(type="pipeline")

st.header("Select a pipeline")
pipeline_id = st.selectbox(
    "Select a pipeline",
    [pipeline.id for pipeline in pipelines]
)
selected_pipeline = automl.registry.get(pipeline_id)

st.write(f"Selected pipeline: {selected_pipeline.name}")

for metadata in selected_pipeline.metadata:
    st.write(f"- {metadata}: {selected_pipeline.metadata[metadata]}")

# change pipeline back to Pipeline object
df = pd.read_csv(io.BytesIO(selected_pipeline.data))

dataset = Dataset.from_dataframe(
    df,
    name=selected_pipeline.name,
    asset_path=selected_pipeline.asset_path,
    version=selected_pipeline.version
)
features = detect_feature_types(dataset)
for feature in features:
    if feature.name == selected_pipeline.metadata["target_feature"]:
        target_feature = feature
        features.remove(feature)
model = get_model(selected_pipeline.metadata["model"])
metrics = []
metrics_list = selected_pipeline.metadata["metrics"].strip("[]").split(", ")
for metric in metrics_list:
    metric = metric.strip("'")
    metrics.append(get_metric(metric))

pipeline = Pipeline(
    dataset=dataset,
    model=model,
    input_features=features,
    target_feature=target_feature,
    metrics=metrics
)
st.write(pipeline)

st.write("You selected this pipeline.")
st.header("Make predictions")
st.write(
    "Upload a CSV file with data in the same manner as the pipeline "
    "to make predictions."
)

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    pipeline._dataset = Dataset.from_dataframe(
        df,
        name=selected_pipeline.name,
        asset_path=selected_pipeline.asset_path,
        version=selected_pipeline.version
    )
    predictions = pipeline.execute()
    st.write(predictions)
