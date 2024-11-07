"""This module contains the instructions page for the Streamlit app."""
import streamlit as st

st.set_page_config(
    page_title="Instructions",
    page_icon="👋",
)

st.markdown(open("INSTRUCTIONS.md").read())
