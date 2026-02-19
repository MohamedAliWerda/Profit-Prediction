"""
app.py
------
Single-file Streamlit app – Startup Profit Prediction System.
Run with:
    streamlit run app.py
"""

import os
import io
import json
import pickle
import random
import smtplib
import string
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import bcrypt
import jwt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import statsmodels.api as sm
from dotenv import load_dotenv

os.chdir(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

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
# AUTH – CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

_USERS_FILE   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.json")
_SMTP_EMAIL   = os.getenv("SMTP_EMAIL", "")
_SMTP_PASS    = os.getenv("SMTP_PASSWORD", "")
_JWT_SECRET   = os.getenv("JWT_SECRET", "changeme_secret")
_JWT_ALGO     = "HS256"
_JWT_EXPIRE   = 3600        # 1 hour
_CODE_EXPIRE  = 600         # 10 minutes


# ════════════════════════════════════════════════════════════════════════════
# AUTH – JSON USER STORE
# ════════════════════════════════════════════════════════════════════════════

def _load_users() -> dict:
    if os.path.exists(_USERS_FILE):
        with open(_USERS_FILE, "r") as f:
            return json.load(f)
    return {}


def _save_users(users: dict) -> None:
    with open(_USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


# ════════════════════════════════════════════════════════════════════════════
# AUTH – EMAIL HELPER
# ════════════════════════════════════════════════════════════════════════════

def _send_email(to_addr: str, subject: str, html_body: str) -> bool:
    """Send an HTML email via Gmail SMTP. Returns True on success."""
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = _SMTP_EMAIL
        msg["To"]      = to_addr
        msg.attach(MIMEText(html_body, "html"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=15) as srv:
            srv.login(_SMTP_EMAIL, _SMTP_PASS)
            srv.sendmail(_SMTP_EMAIL, to_addr, msg.as_string())
        return True
    except Exception as exc:
        st.error(f"❌ Could not send email: {exc}")
        return False


def _make_code() -> str:
    """Generate a 6-digit numeric OTP."""
    return "".join(random.choices(string.digits, k=6))


# ════════════════════════════════════════════════════════════════════════════
# AUTH – JWT
# ════════════════════════════════════════════════════════════════════════════

def _create_jwt(email: str) -> str:
    payload = {
        "sub": email,
        "iat": int(time.time()),
        "exp": int(time.time()) + _JWT_EXPIRE,
    }
    return jwt.encode(payload, _JWT_SECRET, algorithm=_JWT_ALGO)


def _verify_jwt(token: str) -> str | None:
    """Return the email from a valid JWT, or None if invalid/expired."""
    try:
        data = jwt.decode(token, _JWT_SECRET, algorithms=[_JWT_ALGO])
        return data["sub"]
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


# ════════════════════════════════════════════════════════════════════════════
# AUTH – CORE ACTIONS
# ════════════════════════════════════════════════════════════════════════════

def _register_user(email: str, password: str) -> tuple[bool, str]:
    users = _load_users()
    if email in users:
        return False, "An account with this email already exists."
    pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    code    = _make_code()
    users[email] = {
        "password_hash":     pw_hash,
        "is_verified":       False,
        "verification_code": code,
        "code_expires":      time.time() + _CODE_EXPIRE,
        "twofa_code":        "",
        "twofa_expires":     0.0,
    }
    _save_users(users)
    html = f"""
    <h2>Verify your email</h2>
    <p>Welcome! Use the code below to verify your account:</p>
    <h1 style="letter-spacing:8px;color:#4CAF50;">{code}</h1>
    <p>This code expires in <b>10 minutes</b>.</p>
    """
    if not _send_email(email, "Startup Predictor – Verify your email", html):
        return False, "Account created but verification email failed to send."
    return True, "Account created! Check your inbox for the verification code."


def _verify_email_code(email: str, code: str) -> tuple[bool, str]:
    users = _load_users()
    if email not in users:
        return False, "Account not found."
    u = users[email]
    if u["is_verified"]:
        return True, "Already verified."
    if time.time() > u["code_expires"]:
        return False, "Code expired. Please request a new one."
    if u["verification_code"] != code.strip():
        return False, "Invalid code."
    users[email]["is_verified"] = True
    _save_users(users)
    return True, "Email verified! You can now log in."


def _resend_verification(email: str) -> tuple[bool, str]:
    users = _load_users()
    if email not in users:
        return False, "Account not found."
    if users[email]["is_verified"]:
        return False, "Account is already verified."
    code = _make_code()
    users[email]["verification_code"] = code
    users[email]["code_expires"]      = time.time() + _CODE_EXPIRE
    _save_users(users)
    html = f"""
    <h2>New verification code</h2>
    <h1 style="letter-spacing:8px;color:#4CAF50;">{code}</h1>
    <p>Expires in <b>10 minutes</b>.</p>
    """
    _send_email(email, "Startup Predictor – New verification code", html)
    return True, "New code sent."


def _login_user(email: str, password: str) -> tuple[bool, str]:
    users = _load_users()
    if email not in users:
        return False, "Invalid email or password."
    u = users[email]
    if not bcrypt.checkpw(password.encode(), u["password_hash"].encode()):
        return False, "Invalid email or password."
    if not u["is_verified"]:
        return False, "EMAIL_NOT_VERIFIED"
    code = _make_code()
    users[email]["twofa_code"]    = code
    users[email]["twofa_expires"] = time.time() + _CODE_EXPIRE
    _save_users(users)
    html = f"""
    <h2>Your login code</h2>
    <p>Use this code to complete sign-in:</p>
    <h1 style="letter-spacing:8px;color:#2196F3;">{code}</h1>
    <p>Expires in <b>10 minutes</b>.</p>
    """
    _send_email(email, "Startup Predictor – Login verification code", html)
    return True, "2FA code sent to your email."


def _verify_twofa(email: str, code: str) -> tuple[bool, str]:
    users = _load_users()
    if email not in users:
        return False, "Account not found."
    u = users[email]
    if time.time() > u["twofa_expires"]:
        return False, "Code expired. Please log in again."
    if u["twofa_code"] != code.strip():
        return False, "Invalid code."
    users[email]["twofa_code"]    = ""
    users[email]["twofa_expires"] = 0.0
    _save_users(users)
    token = _create_jwt(email)
    return True, token


# ════════════════════════════════════════════════════════════════════════════
# AUTH – STREAMLIT PAGES
# ════════════════════════════════════════════════════════════════════════════

def _init_auth_state() -> None:
    defaults = {
        "auth_jwt":           "",
        "auth_step":          "login",   # login | register | verify_email | two_fa
        "auth_pending_email": "",
        "authenticated":      False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _check_existing_jwt() -> bool:
    """If a valid JWT is in session, mark user as authenticated."""
    if st.session_state.get("auth_jwt"):
        email = _verify_jwt(st.session_state["auth_jwt"])
        if email:
            st.session_state["authenticated"] = True
            return True
        else:
            st.session_state["auth_jwt"]      = ""
            st.session_state["authenticated"] = False
    return False


def render_auth() -> bool:
    """
    Render the authentication wall.
    Returns True when the user is authenticated (main app should render).
    """
    _init_auth_state()

    if _check_existing_jwt():
        return True

    step = st.session_state["auth_step"]

    # ── centred card ──────────────────────────────────────────────────────
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown("## 💰 Startup Profit Predictor")
        st.markdown("---")

        if step == "login":
            _auth_login_page()
        elif step == "register":
            _auth_register_page()
        elif step == "verify_email":
            _auth_verify_email_page()
        elif step == "two_fa":
            _auth_twofa_page()

    return False


# -- individual step pages -----------------------------------------------------

def _auth_login_page() -> None:
    st.markdown("### 🔐 Sign In")
    email    = st.text_input("Email", key="li_email")
    password = st.text_input("Password", type="password", key="li_pass")

    if st.button("Sign In", type="primary", use_container_width=True):
        if not email or not password:
            st.warning("Please fill in all fields.")
        else:
            ok, msg = _login_user(email, password)
            if ok:
                st.session_state["auth_pending_email"] = email
                st.session_state["auth_step"]          = "two_fa"
                st.success(msg)
                st.rerun()
            elif msg == "EMAIL_NOT_VERIFIED":
                st.error("Your email is not verified. Please verify it first.")
                st.session_state["auth_pending_email"] = email
                if st.button("Resend verification code"):
                    _resend_verification(email)
                    st.session_state["auth_step"] = "verify_email"
                    st.rerun()
            else:
                st.error(msg)

    st.markdown("---")
    st.markdown("Don't have an account?")
    if st.button("Create account", use_container_width=True):
        st.session_state["auth_step"] = "register"
        st.rerun()


def _auth_register_page() -> None:
    st.markdown("### 📝 Create Account")
    email    = st.text_input("Email", key="reg_email")
    password = st.text_input("Password (min 8 chars)", type="password", key="reg_pass")
    confirm  = st.text_input("Confirm Password",       type="password", key="reg_conf")

    if st.button("Create Account", type="primary", use_container_width=True):
        if not email or not password or not confirm:
            st.warning("Please fill in all fields.")
        elif "@" not in email:
            st.warning("Please enter a valid email address.")
        elif len(password) < 8:
            st.warning("Password must be at least 8 characters.")
        elif password != confirm:
            st.error("Passwords do not match.")
        else:
            with st.spinner("Creating account & sending verification email…"):
                ok, msg = _register_user(email, password)
            if ok:
                st.session_state["auth_pending_email"] = email
                st.session_state["auth_step"]          = "verify_email"
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    st.markdown("---")
    if st.button("Back to Sign In", use_container_width=True):
        st.session_state["auth_step"] = "login"
        st.rerun()


def _auth_verify_email_page() -> None:
    email = st.session_state.get("auth_pending_email", "")
    st.markdown("### ✉️ Verify Your Email")
    st.info(f"A 6-digit code was sent to **{email}**. Enter it below.")

    code = st.text_input("Verification Code", max_chars=6, key="ver_code")

    if st.button("Verify", type="primary", use_container_width=True):
        if len(code.strip()) != 6:
            st.warning("Please enter the 6-digit code.")
        else:
            ok, msg = _verify_email_code(email, code)
            if ok:
                st.success(msg)
                st.session_state["auth_step"] = "login"
                st.rerun()
            else:
                st.error(msg)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Resend code", use_container_width=True):
            with st.spinner("Sending…"):
                ok, msg = _resend_verification(email)
            if ok:
                st.success(msg)
            else:
                st.error(msg)
    with col2:
        if st.button("Back to Sign In", use_container_width=True):
            st.session_state["auth_step"] = "login"
            st.rerun()


def _auth_twofa_page() -> None:
    email = st.session_state.get("auth_pending_email", "")
    st.markdown("### 🔑 Two-Factor Authentication")
    st.info(f"A 6-digit login code was sent to **{email}**.")

    code = st.text_input("Enter code", max_chars=6, key="tfa_code")

    if st.button("Confirm", type="primary", use_container_width=True):
        if len(code.strip()) != 6:
            st.warning("Please enter the 6-digit code.")
        else:
            ok, token_or_msg = _verify_twofa(email, code)
            if ok:
                st.session_state["auth_jwt"]      = token_or_msg
                st.session_state["authenticated"] = True
                st.success("✅ Signed in successfully!")
                st.rerun()
            else:
                st.error(token_or_msg)

    st.markdown("---")
    if st.button("Back to Sign In", use_container_width=True):
        st.session_state["auth_step"] = "login"
        st.rerun()


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
# BACKWARD ELIMINATION HELPER
# ════════════════════════════════════════════════════════════════════════════

def _clean_feat_name(name: str) -> str:
    """Shorten statsmodels column-transformer prefixes for readability."""
    name = name.replace("remainder__", "").replace("state_encoder__x0_", "State_")
    return name


def backward_elimination_steps(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    sl: float = 0.05,
) -> tuple[list[dict], object]:
    """
    Exact replica of the user's backward_elimination_array logic.
    Works on raw column indices exactly like the original notebook code.
    """
    num_vars         = X.shape[1]
    selected_columns = list(range(num_vars))   # e.g. [0,1,2,3,4,5]
    steps            = []
    iteration        = 0
    regressor_OLS    = None

    while True:
        iteration += 1

        X_opt         = X[:, selected_columns]
        regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
        p_values      = regressor_OLS.pvalues   # len == len(selected_columns)

        cur_names     = [feature_names[i] for i in selected_columns]
        max_p_value   = float(p_values[1:].max())   # skip const at index 0

        step = {
            "iteration":       iteration,
            "selected_columns": list(selected_columns),
            "selected_names":  cur_names,
            "p_values":        dict(zip(cur_names, [float(v) for v in p_values])),
            "max_p":           max_p_value,
            "removed":         None,
            "significant":     max_p_value <= sl,
        }

        if max_p_value > sl:
            # np.argmax over p_values[1:] gives local index; +1 to include const offset
            max_p_index = int(np.argmax(p_values[1:])) + 1
            step["removed"] = cur_names[max_p_index]
            steps.append(step)
            selected_columns.pop(max_p_index)
        else:
            steps.append(step)
            break

    return steps, regressor_OLS


def _step_to_df(step: dict) -> pd.DataFrame:
    """Convert one backward-elimination step to a display DataFrame."""
    rows = []
    for feat, pval in step["p_values"].items():
        rows.append({
            "Feature":    feat,
            "P-Value":    round(pval, 6),
            "Sig (0.05)": "✅" if pval < 0.05 else "❌",
            "Action":     "🗑️ REMOVED" if feat == step["removed"] else "",
        })
    return pd.DataFrame(rows)


def _final_summary_df(model, feature_names: list[str]) -> pd.DataFrame:
    """Build a clean summary table from the final OLS model (numpy-array safe)."""
    params = np.asarray(model.params)
    pvals  = np.asarray(model.pvalues)
    bse    = np.asarray(model.bse)
    tvals  = np.asarray(model.tvalues)
    conf   = np.asarray(model.conf_int())   # shape (k, 2)
    rows   = []
    for i, feat in enumerate(feature_names):
        rows.append({
            "Feature":     feat,
            "Coefficient": round(float(params[i]),     4),
            "Std Error":   round(float(bse[i]),        4),
            "t-Statistic": round(float(tvals[i]),      4),
            "P-Value":     round(float(pvals[i]),      6),
            "95% CI Low":  round(float(conf[i, 0]),    4),
            "95% CI High": round(float(conf[i, 1]),    4),
            "Sig (0.05)":  "✅" if float(pvals[i]) < 0.05 else "❌",
        })
    return pd.DataFrame(rows)



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
# PAGE: BACKWARD ELIMINATION
# ════════════════════════════════════════════════════════════════════════════

def render_backward_elimination(models: dict) -> None:
    st.title("🔬 Backward Elimination")
    st.markdown(
        "Re-run the automated backward elimination on the full feature set "
        "and inspect every step."
    )

    if models["full"] is None:
        st.error("❌ Full model (startup_model_full.pkl) is not loaded.")
        return

    st.markdown("---")

    col_btn1, col_btn2, _ = st.columns([1, 1, 2])
    show_final = col_btn1.button("📋 Show Final Result", key="be_final", use_container_width=True)
    show_steps = col_btn2.button("📊 Show All Steps",    key="be_steps", use_container_width=True)

    # persist results across reruns with session_state
    if "be_steps_data" not in st.session_state:
        st.session_state["be_steps_data"] = None
        st.session_state["be_final_model"] = None

    if show_final or show_steps:
        X_train       = models["full"].model.exog
        y_train       = models["full"].model.endog
        raw_names     = list(models["full"].model.exog_names)
        feature_names = [_clean_feat_name(n) for n in raw_names]

        with st.spinner("Running backward elimination…"):
            steps, final_model = backward_elimination_steps(
                X_train, y_train, feature_names, sl=0.05
            )
        st.session_state["be_steps_data"]  = steps
        st.session_state["be_final_model"] = final_model
        st.session_state["be_show"]        = "final" if show_final else "steps"

    steps       = st.session_state.get("be_steps_data")
    final_model = st.session_state.get("be_final_model")
    show_mode   = st.session_state.get("be_show", "")

    if steps is None:
        st.info("Click a button above to run the backward elimination.")
        return

    # ── Final result ──────────────────────────────────────────────────────
    if show_mode == "final" or show_final:
        last_step = steps[-1]
        st.markdown("#### 📋 Final Model – Coefficient Table")
        m1, m2, m3 = st.columns(3)
        m1.metric("R² Score",       f"{final_model.rsquared:.4f}")
        m2.metric("Adj. R²",        f"{final_model.rsquared_adj:.4f}")
        m3.metric("Remaining vars",  len(last_step["selected_names"]) - 1)
        final_df = _final_summary_df(final_model, last_step["selected_names"])
        st.dataframe(
            final_df.style
                .format(
                    {"Coefficient": "{:.4f}", "Std Error": "{:.4f}",
                     "t-Statistic": "{:.4f}", "P-Value": "{:.6f}",
                     "95% CI Low":  "{:.4f}", "95% CI High": "{:.4f}"}
                ),
            use_container_width=True,
        )

    # ── All steps ─────────────────────────────────────────────────────────
    if show_mode == "steps" or show_steps:
        st.markdown("#### 📊 All Backward Elimination Steps")
        for step in steps:
            removed_txt = (
                f" → removed **{step['removed']}**"
                if step["removed"]
                else " → **✅ all significant – STOP**"
            )
            with st.expander(
                f"Iteration {step['iteration']}  |  "
                f"Max p = {step['max_p']:.6f}{removed_txt}",
                expanded=(step is steps[-1]),
            ):
                step_df = _step_to_df(step)
                st.dataframe(
                    step_df.style
                        .format({"P-Value": "{:.6f}"}),
                    use_container_width=True,
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

    # ── ROI calculations ──────────────────────────────────────────────────
    if is_simple:
        results["Total Investment ($)"] = results["R&D Spend"]
    else:
        results["Total Investment ($)"] = (
            results["R&D Spend"] + results["Administration"] + results["Marketing Spend"]
        )

    results["ROI ($)"] = results["Predicted Profit"] - results["Total Investment ($)"]
    results["ROI (%)"] = (
        results["ROI ($)"] / results["Total Investment ($)"] * 100
    )

    st.markdown("---")
    st.markdown("### 📊 Prediction Results")
    st.dataframe(
        results.style.format({
            "Predicted Profit":    "${:,.2f}",
            "Total Investment ($)": "${:,.2f}",
            "ROI ($)":             "${:,.2f}",
            "ROI (%)":             "{:.1f}%",
        }),
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

# ── Authentication gate ───────────────────────────────────────────────────
if not render_auth():
    st.stop()

# ── Authenticated area ────────────────────────────────────────────────────
models = load_models()

with st.sidebar:
    st.title("💰 Startup Profit Predictor")
    st.markdown("---")

    # Show logged-in user and logout
    jwt_email = _verify_jwt(st.session_state.get("auth_jwt", ""))
    if jwt_email:
        st.caption(f"Signed in as **{jwt_email}**")
    if st.button("🚪 Sign Out", use_container_width=True):
        st.session_state["auth_jwt"]      = ""
        st.session_state["authenticated"] = False
        st.session_state["auth_step"]     = "login"
        st.rerun()

    st.markdown("---")
    page = st.radio(
        "Navigate",
        [
            "⚖️ Model Comparison",
            "� Backward Elimination",
            "�🔮 Predictions",
            "📂 CSV Batch Prediction",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")

if page == "⚖️ Model Comparison":
    render_model_comparison(models)
elif page == "� Backward Elimination":
    render_backward_elimination(models)
elif page == "�🔮 Predictions":
    render_predictions(models)
elif page == "📂 CSV Batch Prediction":
    render_csv_batch(models)
