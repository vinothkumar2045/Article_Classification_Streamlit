# ==========================================
# FAST STREAMLIT ARTICLE CLASSIFICATION APP
# WITH RDS AUTHENTICATION (READY TO RUN)
# ==========================================

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import hashlib

from joblib import load
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sqlalchemy import create_engine, text

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(page_title="Article Classification System", layout="centered")
st.title("📰 Article Classification System")

# =============================
# DATABASE CONFIG (RDS)
# =============================
DB_HOST = "streamlit-db.c10c0s8c6mju.ap-south-1.rds.amazonaws.com"
DB_PORT = "5432"
DB_NAME = "articledb"
DB_USER = "dbadmin"
DB_PASSWORD = "cpvinoth2045"   # 🔐 already confirmed

@st.cache_resource
def get_db_engine():
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
    query = text("""
        INSERT INTO users (username, password)
        VALUES (:u, :p)
    """)
    with get_db_engine().begin() as conn:
        conn.execute(query, {
            "u": username,
            "p": hash_password(password)
        })

def validate_login(username, password):
    query = text("""
        SELECT id FROM users
        WHERE username = :u AND password = :p
    """)
    with get_db_engine().connect() as conn:
        return conn.execute(query, {
            "u": username,
            "p": hash_password(password)
        }).fetchone() is not None

def log_prediction(username, model_used):
    query = text("""
        INSERT INTO user_logins (username, model_used)
        VALUES (:u, :m)
    """)
    with get_db_engine().begin() as conn:
        conn.execute(query, {"u": username, "m": model_used})

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

page = st.sidebar.radio(
    "Go to",
    pages,
    index=pages.index(st.session_state.page)
)

st.session_state.page = page

# LOGOUT BUTTON
if st.session_state.logged_in:
    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.page = "Login"
        st.rerun()

# BLOCK ACCESS
if not st.session_state.logged_in and page != "Login":
    st.warning("Please login first")
    st.stop()

# =============================
# LOAD MODELS (FAST & CACHED)
# =============================
@st.cache_resource
def ml_model():
    return load("models/best_model_LogisticRegression.joblib")

@st.cache_resource
def tokenizer():
    with open("models/tokenizer.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def lstm_model():
    return load_model("models/lstm_text_classifier.h5")

@st.cache_resource
def gru_model():
    return load_model("models/gru_text_classifier.h5")

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

    model_type = st.selectbox(
        "Select Model",
        ["Logistic Regression", "LSTM", "GRU"]
    )

    if st.button("Predict"):
        if not article.strip():
            st.warning("Please enter article text")
            st.stop()

        if model_type == "Logistic Regression":
            proba = ml_model().predict_proba([article])[0]

        elif model_type == "LSTM":
            seq = pad_sequences(
                tokenizer().texts_to_sequences([article]),
                maxlen=200
            )
            proba = lstm_model().predict(seq, verbose=0)[0]

        else:
            seq = pad_sequences(
                tokenizer().texts_to_sequences([article]),
                maxlen=150
            )
            proba = gru_model().predict(seq, verbose=0)[0]

        pred_idx = int(np.argmax(proba)) + 1
        pred_label = label_map[pred_idx]
        confidence = np.max(proba) * 100

        st.success(f"Predicted Category: **{pred_label}**")
        st.info(f"Confidence: **{confidence:.2f}%**")

        # 📊 CATEGORY PROBABILITIES
        st.subheader("📊 Category-wise Probability")

        prob_df = pd.DataFrame({
            "Category": list(label_map.values()),
            "Probability (%)": [round(p * 100, 2) for p in proba]
        })

        st.table(prob_df)

        log_prediction(st.session_state.username, model_type)

# =============================
# PAGE 3: MODEL COMPARISON
# =============================
elif page == "Model Comparison":
    st.dataframe(
        pd.DataFrame({
            "Model": ["Logistic Regression", "LSTM", "GRU"],
            "Speed": ["Very Fast", "Fast", "Fast"],
            "Accuracy (%)": [90.6, 92.8, 94.1]
        }),
        use_container_width=True
    )
