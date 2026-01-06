# ==========================================
# FAST & STABLE ARTICLE CLASSIFICATION APP
# (LR + LSTM + GRU) WITH RDS AUTH
# ==========================================

import streamlit as st
import numpy as np
import pickle
import pandas as pd
import hashlib

from joblib import load
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sqlalchemy import create_engine, text

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(page_title="Article Classification", layout="centered")
st.title("📰 Article Classification System")

# =============================
# DATABASE CONFIG
# =============================
DB_HOST = "streamlit-db.c10c0s8c6mju.ap-south-1.rds.amazonaws.com"
DB_PORT = "5432"
DB_NAME = "articledb"
DB_USER = "dbadmin"
DB_PASSWORD = "cpvinoth2045"

@st.cache_resource
def get_engine():
    return create_engine(
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
        pool_pre_ping=True
    )

# =============================
# AUTH HELPERS
# =============================
def hash_pwd(pwd):
    return hashlib.sha256(pwd.encode()).hexdigest()

def register_user(u, p):
    with get_engine().begin() as conn:
        conn.execute(
            text("INSERT INTO users(username,password) VALUES(:u,:p)"),
            {"u": u, "p": hash_pwd(p)}
        )

def validate_login(u, p):
    with get_engine().connect() as conn:
        r = conn.execute(
            text("SELECT id FROM users WHERE username=:u AND password=:p"),
            {"u": u, "p": hash_pwd(p)}
        ).fetchone()
        return r is not None

# =============================
# SESSION STATE
# =============================
if "login" not in st.session_state:
    st.session_state.login = False
if "user" not in st.session_state:
    st.session_state.user = ""
if "page" not in st.session_state:
    st.session_state.page = "Login"

# =============================
# SIDEBAR
# =============================
st.sidebar.title("📌 Navigation")
pages = ["Login", "Article Classification", "Model Comparison"]
page = st.sidebar.radio("Go to", pages, index=pages.index(st.session_state.page))
st.session_state.page = page

if st.session_state.login:
    if st.sidebar.button("🚪 Logout"):
        st.session_state.login = False
        st.session_state.user = ""
        st.session_state.page = "Login"
        st.rerun()

if not st.session_state.login and page != "Login":
    st.warning("Please login first")
    st.stop()

# =============================
# LOAD MODELS (CACHED = FAST)
# =============================
@st.cache_resource(show_spinner="Loading Logistic Regression...")
def lr_model():
    return load("models/best_model_LogisticRegression.joblib")

@st.cache_resource(show_spinner="Loading tokenizer...")
def text_tokenizer():
    with open("models/tokenizer.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource(show_spinner="Loading LSTM...")
def lstm_model():
    return load_model("models/lstm_text_classifier.h5")

@st.cache_resource(show_spinner="Loading GRU...")
def gru_model():
    return load_model("models/gru_text_classifier.h5")

# ✅ SINGLE CORRECT LABEL ORDER
LABELS = ["World", "Sports", "Business", "Technology"]

# =============================
# LOGIN PAGE
# =============================
if page == "Login":
    st.subheader("🔐 Authentication")
    mode = st.radio("Choose", ["Login", "Register"], horizontal=True)

    if mode == "Login":
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")

        if st.button("Login"):
            if validate_login(u, p):
                st.session_state.login = True
                st.session_state.user = u
                st.session_state.page = "Article Classification"
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid credentials")

    else:
        ru = st.text_input("New Username")
        rp = st.text_input("New Password", type="password")

        if st.button("Register"):
            try:
                register_user(ru, rp)
                st.success("Registered successfully. Login now.")
            except:
                st.error("Username already exists")

# =============================
# ARTICLE CLASSIFICATION
# =============================
elif page == "Article Classification":
    st.success(f"Logged in as: {st.session_state.user}")

    article = st.text_area("Enter Article Text", height=200)
    model_choice = st.selectbox("Select Model", ["Logistic Regression", "LSTM", "GRU"])

    if st.button("Predict") and article.strip():

        if model_choice == "Logistic Regression":
            probs = lr_model().predict_proba([article])[0]

        elif model_choice == "LSTM":
            seq = pad_sequences(
                text_tokenizer().texts_to_sequences([article]),
                maxlen=200
            )
            probs = lstm_model().predict(seq, verbose=0)[0]

        else:  # GRU
            seq = pad_sequences(
                text_tokenizer().texts_to_sequences([article]),
                maxlen=150
            )
            probs = gru_model().predict(seq, verbose=0)[0]

        pred_idx = int(np.argmax(probs))
        st.success(f"✅ Predicted Category: **{LABELS[pred_idx]}**")
        st.info(f"📊 Confidence: **{probs[pred_idx]*100:.2f}%**")

        df = pd.DataFrame({
            "Category": LABELS,
            "Probability (%)": np.round(probs * 100, 2)
        })
        st.subheader("📊 Category-wise Probability")
        st.dataframe(df, use_container_width=True)

# =============================
# MODEL COMPARISON
# =============================
elif page == "Model Comparison":
    st.dataframe(pd.DataFrame({
        "Model": ["Logistic Regression", "LSTM", "GRU"],
        "Accuracy (%)": [90.6, 92.8, 94.1],
        "Speed": ["⚡ Very Fast", "Fast", "Fast"]
    }), use_container_width=True)
