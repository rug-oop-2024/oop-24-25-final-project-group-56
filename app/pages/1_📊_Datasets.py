"""This module provides a Streamlit interface for managing datasets."""
import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

# your code here
st.title("Datasets")

st.header("List of saved datasets")
for dataset in datasets:
    st.write(f"- {dataset.name}")

st.header("Upload a CSV file")

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    name = uploaded_file.name
    asset_path = f"assets/datasets/{name}"
    dataset = Dataset.from_dataframe(df, name, asset_path, version="1.0.0")

    st.header("Save the dataset")
    if st.button("Save"):
        automl.registry.register(dataset)
        st.success("Dataset saved successfully")
