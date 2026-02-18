"""
app.py
------
Main entry point.  Run with:
    streamlit run app.py
"""

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from utils.model_loader import load_models
import views.home
import views.model_comparison
import views.predictions
import views.csv_batch

st.set_page_config(
    page_title="Startup Profit Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .big-font {
        font-size: 20px !important;
        font-weight: bold;
    }
    .model-card {
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #f0f0f0;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

models = load_models()

with st.sidebar:
    st.title("💰 Startup Profit Predictor")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        [
            "🏠 Home",
            "⚖️ Model Comparison",
            "🔮 Predictions",
            "📂 CSV Batch Prediction",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")

if page == "🏠 Home":
    views.home.render(models)
elif page == "⚖️ Model Comparison":
    views.model_comparison.render(models)
elif page == "🔮 Predictions":
    views.predictions.render(models)
elif page == "📂 CSV Batch Prediction":
    views.csv_batch.render(models)
