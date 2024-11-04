import streamlit as st
import pandas as pd
import io
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.metric import get_metric
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.model import get_model
from autoop.core.ml.artifact import Artifact

st.set_page_config(page_title="Modelling", page_icon="📈")

def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)

st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine learning pipeline to train a model on a dataset.")

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

# your code here

st.header("Select a dataset")
dataset_id = st.selectbox("Select a dataset", [dataset.id for dataset in datasets])

dataset = automl.registry.get(dataset_id)

st.write(f"Selected dataset: {dataset.name}")

df = pd.read_csv(io.BytesIO(dataset.data))

dataset = Dataset.from_dataframe(df, name=dataset.name, asset_path=dataset.asset_path, version=dataset.version)
features = detect_feature_types(dataset)

for feature in features:
    st.write(f"- {feature.name}: {feature.type}")

st.header("Select the target feature")
target_feature = st.selectbox("Select the target feature", [feature for feature in features])
input_features = st.multiselect("Select the input features", [feature for feature in features if feature != target_feature])

st.header("Select the model")
if target_feature.type == "categorical":
    model_str = st.selectbox("Select the model", ["LogisticRegression", "MLPClassifier", "RidgeClassifier"])
else:
    model_str = st.selectbox("Select the model", ["MultipleLinearRegression", "Lasso", "MLPRegressor"])

model = get_model(model_str)



st.write(f"Selected model: {model_str}")
st.header("Select a dataset split")
split_ratio = st.slider("Select the split ratio", min_value=0.1, max_value=0.9, value=0.8, step=0.1)
st.write(f"Selected split ratio for training and testing: {split_ratio}")

train_size = int((len(dataset.data) * split_ratio))
train_data = dataset.data[:train_size]
test_data = dataset.data[train_size:]

st.write("Train data:")
st.write(train_data)

st.write("Test data:")
st.write(test_data)

#Select metrics
st.header("Select metrics")
if target_feature.type == "categorical":
    metrics = st.multiselect("Select metrics", ["accuracy", "precision", "recall"])
else:
    metrics = st.multiselect("Select metrics", ["mean_squared_error", "mean_absolute_error", "r_squared"])

metrics_use = []
st.write("Selected metrics:")
for metric in metrics:
    st.write(f"- {metric}")
    metrics_use.append(get_metric(metric))

#Summary of choices
st.header("Summary of choices")
st.write(f"Selected dataset: {dataset.name}")
st.write(f"Selected target feature: {target_feature.name}")
st.write(f"Selected model: {model_str}")
st.write(f"Selected split ratio: {split_ratio}")
st.write("Selected metrics:")
for metric in metrics:
    st.write(f"- {metric}")

#Train model

def execute_pipeline():
    pipeline = Pipeline(metrics_use, dataset, model, input_features, target_feature, split_ratio)
    pipeline.execute()
    return pipeline
st.header("Train model")
if st.button("Train"):
    pipeline = execute_pipeline()
    st.write("Model trained successfully")
    #Print model
    st.header("Results of training")
    st.write("Predictions:")
    st.write(pipeline._predictions)
    st.write("Metrics results:")
    for metric, result in pipeline._metrics_results:
        st.write(f"- {metric}: {result}")

#Save pipeline
st.header("Save pipeline")
pipeline_name = st.text_input("Enter a name for the pipeline")
if st.button("Save"):
    pipeline = execute_pipeline()
    # artifacts = pipeline.artifacts
    pipeline_artifact = Artifact(name=pipeline_name, type="pipeline", asset_path=f"assets/pipelines/{pipeline_name}", data=pipeline._dataset.data)
    pipeline_artifact.id = pipeline_artifact.generate_id()
    #for artifact in artifacts:
        #artifact.name = pipeline_name
        #artifact.asset_path = f"assets/pipelines/{pipeline_name}"
        #artifact.id = artifact.generate_id()
    automl.registry.register(pipeline_artifact)
    st.write("Pipeline saved successfully")
    
