# ==========================================
# FAST & STABLE STREAMLIT ARTICLE CLASSIFIER
# Logistic Regression (Production Ready)
# With RDS Login / Register / Logout
# ==========================================

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import hashlib

from joblib import load
from sqlalchemy import create_engine, text

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Article Classification System",
    layout="centered"
)

st.title("📰 Article Classification System")

# =============================
# DATABASE CONFIG (RDS)
# =============================
DB_HOST = "streamlit-db.c10c0s8c6mju.ap-south-1.rds.amazonaws.com"
DB_PORT = "5432"
DB_NAME = "articledb"
DB_USER = "dbadmin"
DB_PASSWORD = "cpvinoth2045"   # your password

@st.cache_resource
def get_engine():
    return create_engine(
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
        pool_pre_ping=True
    )

# =============================
# AUTH HELPERS
# =============================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    q = text("""
        INSERT INTO users (username, password)
        VALUES (:u, :p)
    """)
    with get_engine().begin() as conn:
        conn.execute(q, {"u": username, "p": hash_password(password)})

def validate_login(username, password):
    q = text("""
        SELECT id FROM users
        WHERE username = :u AND password = :p
    """)
    with get_engine().connect() as conn:
        return conn.execute(
            q, {"u": username, "p": hash_password(password)}
        ).fetchone() is not None

# =============================
# SESSION STATE
# =============================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "page" not in st.session_state:
    st.session_state.page = "Login"

# =============================
# SIDEBAR NAVIGATION
# =============================
st.sidebar.title("📌 Navigation")

pages = ["Login", "Article Classification", "Model Comparison"]
page = st.sidebar.radio("Go to", pages, index=pages.index(st.session_state.page))
st.session_state.page = page

# Logout button
if st.session_state.logged_in:
    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.page = "Login"
        st.rerun()

# Block access if not logged in
if not st.session_state.logged_in and page != "Login":
    st.warning("Please login first")
    st.stop()

# =============================
# LOAD MODEL (FAST)
# =============================
@st.cache_resource
def load_ml_model():
    return load("models/best_model_LogisticRegression.joblib")

label_map = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Technology"
}

# =============================
# PAGE 1: LOGIN / REGISTER
# =============================
if page == "Login":
    st.subheader("🔐 User Authentication")

    mode = st.radio("Select Action", ["Login", "Register"], horizontal=True)

    if mode == "Login":
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")

        if st.button("Login"):
            if validate_login(u, p):
                st.session_state.logged_in = True
                st.session_state.username = u
                st.session_state.page = "Article Classification"
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid username or password")

    else:
        ru = st.text_input("New Username")
        rp = st.text_input("New Password", type="password")

        if st.button("Register"):
            try:
                register_user(ru, rp)
                st.success("Registration successful. Please login.")
            except:
                st.error("Username already exists")

# =============================
# PAGE 2: ARTICLE CLASSIFICATION
# =============================
elif page == "Article Classification":
    st.success(f"Logged in as: {st.session_state.username}")

    article = st.text_area("Enter Article Text", height=200)

    if st.button("Predict"):
        model = load_ml_model()
        proba = model.predict_proba([article])[0]

        pred_idx = int(np.argmax(proba)) + 1
        confidence = np.max(proba) * 100

        st.success(f"✅ Predicted Category: **{label_map[pred_idx]}**")
        st.info(f"📊 Confidence: **{confidence:.2f}%**")

        # Category-wise probability table
        prob_df = pd.DataFrame({
            "Category": list(label_map.values()),
            "Probability (%)": (proba * 100).round(2)
        })

        st.subheader("📊 Category-wise Probability")
        st.dataframe(prob_df, use_container_width=True)

# =============================
# PAGE 3: MODEL COMPARISON
# =============================
elif page == "Model Comparison":
    st.subheader("📈 Model Performance Comparison")

    df = pd.DataFrame({
        "Model": ["Logistic Regression", "LSTM", "GRU"],
        "Speed": ["Very Fast", "Slow", "Slow"],
        "Accuracy (%)": [90.6, 92.8, 94.1],
        "Production Use": ["✅ Yes", "❌ No", "❌ No"]
    })

    st.dataframe(df, use_container_width=True)
