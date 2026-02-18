"""
app.py
------
Single-file Streamlit app – Startup Profit Prediction System.
Run with:
    streamlit run app.py
"""

import os
import io
import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import statsmodels.api as sm

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── sklearn 1.7 compatibility ──────────────────────────────────────────────
import sklearn.compose._column_transformer as _ct_module

if not hasattr(_ct_module, "_RemainderColsList"):
    class _RemainderColsList(list):
        """Stub for backwards-compatible unpickling of ColumnTransformer."""
        def __reduce__(self):
            return (list, (list(self),))
    _ct_module._RemainderColsList = _RemainderColsList
# ──────────────────────────────────────────────────────────────────────────


# ════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_models() -> dict:
    models: dict = {}

    try:
        with open("startup_model.pkl", "rb") as f:
            models["simple"] = pickle.load(f)
    except Exception:
        models["simple"] = None

    try:
        with open("startup_model_full.pkl", "rb") as f:
            models["full"] = pickle.load(f)
        with open("column_transformer.pkl", "rb") as f:
            models["ct"] = pickle.load(f)
    except Exception:
        models["full"] = None
        models["ct"] = None

    return models


# ════════════════════════════════════════════════════════════════════════════
# PREDICTION FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def predict_simple(models: dict, rd_spend: float):
    if models["simple"] is None:
        return None
    X_new = sm.add_constant([[rd_spend]], has_constant="add")
    return float(models["simple"].predict(X_new)[0])


def predict_full(models: dict, state: str, rd_spend: float,
                 admin_spend: float, marketing_spend: float):
    if models["full"] is None or models["ct"] is None:
        return None

    state_mapping = {"California": 0, "Florida": 1, "New York": 2}
    state_int = state_mapping[state]
    state_df = pd.DataFrame({"State": [state_int]})

    ohe = models["ct"].named_transformers_["state_encoder"]
    ohe_encoded = ohe.transform(state_df)
    if hasattr(ohe_encoded, "toarray"):
        ohe_encoded = ohe_encoded.toarray()

    ohe_cols = [c for c in models["ct"].get_feature_names_out()
                if c.startswith("state_encoder__")]
    input_df = pd.DataFrame(ohe_encoded, columns=ohe_cols)

    input_df["remainder__R&D Spend"]       = rd_spend
    input_df["remainder__Administration"]  = admin_spend
    input_df["remainder__Marketing Spend"] = marketing_spend

    input_with_const = sm.add_constant(input_df, has_constant="add")
    return float(models["full"].predict(input_with_const)[0])


# ════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ════════════════════════════════════════════════════════════════════════════

def render_home(models: dict) -> None:
    st.title("💰 Startup Profit Prediction System")
    st.markdown("### 🎓 Advanced Model Comparison Platform")

    st.markdown("""
    This application demonstrates **professional Machine Learning workflow**
    by comparing two regression approaches:

    1. **Backward Elimination Method** – Uses only R&D Spend
    2. **All Features Selection Method** – Uses all available data

    ---
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🎯 Backward Elimination Method")
        st.markdown("""
        **Features:**
        - ✅ R&D Spend only

        **Advantages:**
        - Very easy to use
        - Statistically significant
        - High R² score
        - Quick predictions

        **Best For:**
        - Quick estimates
        - When only R&D data is available
        - Simple interpretation
        """)
        if models["simple"]:
            st.success(f"**R² Score:** {models['simple'].rsquared:.4f}")

    with col2:
        st.markdown("#### 🎯 All Features Selection Method")
        st.markdown("""
        **Features:**
        - ✅ R&D Spend
        - ✅ Administration
        - ✅ Marketing Spend
        - ✅ State (Location)

        **Advantages:**
        - Uses all available information
        - More comprehensive
        - Captures all relationships

        **Best For:**
        - Detailed analysis
        - When all data is available
        - Complete picture
        """)
        if models["full"]:
            st.success(f"**R² Score:** {models['full'].rsquared:.4f}")

    st.markdown("---")


# ════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICTIONS
# ════════════════════════════════════════════════════════════════════════════

def render_predictions(models: dict) -> None:
    st.title("🔮 Make Predictions")

    model_choice = st.radio(
        "Choose Model:",
        ["🎯 Backward Elimination Method", "🎯 All Features Selection Method"],
        horizontal=True,
    )

    st.markdown("---")

    if "Backward Elimination" in model_choice:
        _render_backward_elimination(models)
    else:
        _render_all_features(models)


def _render_backward_elimination(models: dict) -> None:
    st.markdown("### 💵 Enter R&D Spend")

    rd_spend = st.number_input(
        "💵 R&D Spend ($)",
        min_value=0, max_value=500_000,
        value=100_000, step=5_000,
    )

    if not st.button("🎯 Predict", type="primary", use_container_width=True):
        return

    profit = predict_simple(models, rd_spend)

    if profit is None:
        st.error("❌ Prediction failed. Check that startup_model.pkl is present.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Predicted Profit", f"${profit:,.2f}")
    col2.metric("📊 ROI",              f"{(profit - rd_spend) / rd_spend * 100:.1f}%")
    col3.metric("📈 Net Gain",         f"${profit - rd_spend:,.2f}")

    rd_range     = np.linspace(0, 300_000, 100)
    profit_range = [predict_simple(models, rd) for rd in rd_range]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rd_range, y=profit_range,
        mode="lines", name="Prediction Curve",
        line=dict(color="blue", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=[rd_spend], y=[profit],
        mode="markers", name="Your Prediction",
        marker=dict(size=15, color="red", symbol="star"),
    ))
    fig.update_layout(
        title="Profit vs R&D Spend",
        xaxis_title="R&D Spend ($)",
        yaxis_title="Predicted Profit ($)",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_all_features(models: dict) -> None:
    st.markdown("### 📝 Complete Startup Profile")

    col1, col2 = st.columns(2)
    with col1:
        rd_spend    = st.number_input("💵 R&D Spend ($)",      value=100_000.0, step=1_000.0)
        admin_spend = st.number_input("🏢 Administration ($)", value=120_000.0, step=1_000.0)
    with col2:
        marketing_spend = st.number_input("📢 Marketing Spend ($)", value=200_000.0, step=1_000.0)
        state           = st.selectbox("📍 State", ["California", "Florida", "New York"])

    if not st.button("🎯 Predict", type="primary", use_container_width=True):
        return

    profit    = predict_full(models, state, rd_spend, admin_spend, marketing_spend)
    total_inv = rd_spend + admin_spend + marketing_spend

    if profit is None:
        st.error("❌ Prediction failed. Check that all model files are present.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Predicted Profit", f"${profit:,.2f}")
    col2.metric("📊 ROI",              f"{(profit - total_inv) / total_inv * 100:.1f}%")
    col3.metric("📈 Net Gain",         f"${profit - total_inv:,.2f}")

    st.markdown("---")
    st.markdown("### 📊 Investment Breakdown")

    fig = go.Figure(data=[go.Pie(
        labels=["R&D", "Administration", "Marketing"],
        values=[rd_spend, admin_spend, marketing_spend],
        hole=0.4,
    )])
    fig.update_layout(title="Investment Distribution", height=400)
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL COMPARISON
# ════════════════════════════════════════════════════════════════════════════

def render_model_comparison(models: dict) -> None:
    st.title("⚖️ Model Comparison")
    st.markdown("### Compare predictions from both models side-by-side")

    if not (models["simple"] and models["full"]):
        st.error("❌ Both models must be loaded to use this page.")
        return

    st.markdown("### 📝 Enter Startup Information")

    col1, col2, col3 = st.columns(3)
    with col1:
        rd_spend = st.number_input(
            "💵 R&D Spend ($)", min_value=0.0, max_value=500_000.0,
            value=100_000.0, step=1_000.0
        )
    with col2:
        admin_spend = st.number_input(
            "🏢 Administration ($)", min_value=0.0, max_value=500_000.0,
            value=120_000.0, step=1_000.0
        )
    with col3:
        marketing_spend = st.number_input(
            "📢 Marketing Spend ($)", min_value=0.0, max_value=500_000.0,
            value=200_000.0, step=1_000.0
        )

    state = st.selectbox("📍 Select State", ["California", "Florida", "New York"])

    st.markdown("---")

    if not st.button("🎯 Compare Predictions", type="primary", use_container_width=True):
        return

    simple_pred = predict_simple(models, rd_spend)
    full_pred   = predict_full(models, state, rd_spend, admin_spend, marketing_spend)

    st.markdown("### 📊 Prediction Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🎯 Backward Elimination Method")
        st.metric("Predicted Profit", f"${simple_pred:,.2f}")
        simple_roi = (simple_pred - rd_spend) / rd_spend * 100
        st.metric("ROI (R&D only)", f"{simple_roi:.1f}%")

    with col2:
        st.markdown("#### 🎯 All Features Selection Method")
        st.metric("Predicted Profit", f"${full_pred:,.2f}")
        total_inv = rd_spend + admin_spend + marketing_spend
        full_roi  = (full_pred - total_inv) / total_inv * 100
        st.metric("ROI (Total)", f"{full_roi:.1f}%")

    with col3:
        st.markdown("#### 📊 Difference")
        diff     = abs(full_pred - simple_pred)
        diff_pct = diff / simple_pred * 100
        st.metric("Absolute Difference",   f"${diff:,.2f}")
        st.metric("Percentage Difference", f"{diff_pct:.1f}%")

    st.markdown("---")
    st.markdown("### 📈 Visual Comparison")

    fig = go.Figure(data=[
        go.Bar(
            name="Backward Elimination Method",
            x=["Prediction"], y=[simple_pred],
            marker_color="lightblue",
            text=[f"${simple_pred:,.2f}"], textposition="outside",
        ),
        go.Bar(
            name="All Features Selection Method",
            x=["Prediction"], y=[full_pred],
            marker_color="lightgreen",
            text=[f"${full_pred:,.2f}"], textposition="outside",
        ),
    ])
    fig.update_layout(
        title="Model Predictions Comparison",
        yaxis_title="Predicted Profit ($)",
        barmode="group",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔍 Analysis")

    if diff_pct < 5:
        st.success(
            "✅ **Models agree closely** (< 5% difference)\n\n"
            "Both models predict similar outcomes. The Backward Elimination Method "
            "may be sufficient for this case."
        )
    elif diff_pct < 10:
        st.info(
            "ℹ️ **Models show moderate difference** (5–10%)\n\n"
            "Consider both predictions. Additional features in the All Features "
            "Selection Method provide some different insights."
        )
    else:
        st.warning(
            "⚠️ **Models show significant difference** (> 10%)\n\n"
            "The additional features (State, Admin, Marketing) significantly impact "
            "the prediction. The All Features model may provide more accurate results."
        )


# ════════════════════════════════════════════════════════════════════════════
# PAGE: CSV BATCH PREDICTION
# ════════════════════════════════════════════════════════════════════════════

_SIMPLE_COLS = ["R&D Spend"]
_FULL_COLS   = ["R&D Spend", "Administration", "Marketing Spend", "State"]


def render_csv_batch(models: dict) -> None:
    st.title("📂 CSV Batch Prediction")
    st.markdown("Upload a CSV file to predict profits for multiple startups at once.")

    with st.expander("📋 Required CSV Format", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Backward Elimination Method")
            st.markdown("Required columns: `R&D Spend`")
            sample_simple = pd.DataFrame({
                "R&D Spend": [142_107.34, 131_876.90, 99_814.71, 162_597.70],
            })
            st.dataframe(sample_simple, use_container_width=True)
            st.download_button(
                "⬇️ Download Simple Template",
                sample_simple.to_csv(index=False),
                "template_simple.csv",
                "text/csv",
            )

        with col2:
            st.markdown("#### All Features Selection Method")
            st.markdown("Required columns: `R&D Spend`, `Administration`, `Marketing Spend`, `State`")
            sample_full = pd.DataFrame({
                "R&D Spend":       [142_107.34, 131_876.90, 99_814.71, 162_597.70],
                "Administration":  [91_391.77,  99_814.71,  60_084.64, 70_258.04],
                "Marketing Spend": [366_168.42, 204_346.76, 280_574.52, 443_898.53],
                "State":           ["New York", "California", "Florida", "New York"],
            })
            st.dataframe(sample_full, use_container_width=True)
            st.download_button(
                "⬇️ Download Full Template",
                sample_full.to_csv(index=False),
                "template_full.csv",
                "text/csv",
            )

    model_choice = st.radio(
        "Choose Model:",
        ["🎯 Backward Elimination Method", "🎯 All Features Selection Method"],
        horizontal=True,
    )
    is_simple = "Backward Elimination" in model_choice

    uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

    if uploaded_file is None:
        return

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"❌ Could not read file: {exc}")
        return

    required = _SIMPLE_COLS if is_simple else _FULL_COLS
    missing  = [c for c in required if c not in df.columns]

    if missing:
        st.error(
            f"❌ The uploaded file is missing required column(s): "
            f"{', '.join(missing)}"
        )
        return

    results = df.copy()

    if is_simple:
        results["Predicted Profit"] = results["R&D Spend"].apply(
            lambda rd: predict_simple(models, rd)
        )
    else:
        def _pred_row(row):
            return predict_full(
                models,
                row["State"],
                row["R&D Spend"],
                row["Administration"],
                row["Marketing Spend"],
            )
        results["Predicted Profit"] = results.apply(_pred_row, axis=1)

    st.markdown("---")
    st.markdown("### 📊 Prediction Results")
    st.dataframe(
        results.style.format({"Predicted Profit": "${:,.2f}"}),
        use_container_width=True,
    )

    valid_preds = results["Predicted Profit"].dropna()
    if len(valid_preds) > 0:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("📊 Total Startups",      len(results))
        m2.metric("💰 Avg Predicted Profit", f"${valid_preds.mean():,.2f}")
        m3.metric("🚀 Highest Profit",       f"${valid_preds.max():,.2f}")
        m4.metric("📉 Lowest Profit",        f"${valid_preds.min():,.2f}")

    st.markdown("---")
    st.markdown("### 📈 Profit Distribution")

    labels = [f"Startup {i+1}" for i in range(len(results))]
    fig = go.Figure(go.Bar(
        x=labels,
        y=results["Predicted Profit"],
        marker_color="steelblue",
        text=results["Predicted Profit"].apply(
            lambda v: f"${v:,.0f}" if pd.notna(v) else "N/A"
        ),
        textposition="outside",
    ))
    fig.update_layout(
        title="Predicted Profits by Startup",
        xaxis_title="Startup",
        yaxis_title="Predicted Profit ($)",
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    csv_bytes = results.to_csv(index=False).encode()
    st.download_button(
        "⬇️ Download Results as CSV",
        data=csv_bytes,
        file_name="predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# APP SETUP & ROUTING
# ════════════════════════════════════════════════════════════════════════════

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
    render_home(models)
elif page == "⚖️ Model Comparison":
    render_model_comparison(models)
elif page == "🔮 Predictions":
    render_predictions(models)
elif page == "📂 CSV Batch Prediction":
    render_csv_batch(models)
