# ==========================================
# FAST STREAMLIT ARTICLE CLASSIFICATION
# WITH RDS AUTH (AWS SAFE)
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
st.set_page_config(page_title="Article Classification System", layout="centered")
st.title("📰 Article Classification System")

# =============================
# DATABASE CONFIG (RDS)
# =============================
DB_HOST = "streamlit-db.c10c0s8c6mju.ap-south-1.rds.amazonaws.com"
DB_PORT = "5432"
DB_NAME = "articledb"
DB_USER = "dbadmin"
DB_PASSWORD = "cpvinoth2045"

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
        WHERE username=:u AND password=:p
    """)
    with get_db_engine().connect() as conn:
        return conn.execute(query, {
            "u": username,
            "p": hash_password(password)
        }).fetchone() is not None

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

# LOGOUT
if st.session_state.logged_in:
    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.page = "Login"
        st.rerun()

# BLOCK UNAUTHENTICATED ACCESS
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
def text_tokenizer():
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
# TEXT CLEANING (IMPORTANT)
# =============================
def clean_text(text):
    text = text.lower()
    text = text.replace("\n", " ")
    return text

# =============================
# PAGE 1: LOGIN / REGISTER
# =============================
if page == "Login":
    st.subheader("🔐 User Authentication")

    auth_mode = st.radio("Select Action", ["Login", "Register"], horizontal=True)

    if auth_mode == "Login":
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

    article = clean_text(
        st.text_area("Enter Article Text", height=200)
    )

    if len(article.split()) < 15:
        st.warning("⚠️ Short text detected. Accuracy may be low.")

    model_type = st.selectbox(
        "Select Model",
        ["Logistic Regression", "LSTM", "GRU"],
        index=0
    )

    if st.button("Predict"):
        if model_type == "Logistic Regression":
            proba = ml_model().predict_proba([article])[0]

        elif model_type == "LSTM":
            seq = pad_sequences(
                text_tokenizer().texts_to_sequences([article]),
                maxlen=200
            )
            proba = lstm_model().predict(seq, verbose=0)[0]

        else:
            seq = pad_sequences(
                text_tokenizer().texts_to_sequences([article]),
                maxlen=150
            )
            proba = gru_model().predict(seq, verbose=0)[0]

        pred_idx = int(np.argmax(proba)) + 1
        st.success(f"Predicted Category: {label_map[pred_idx]}")
        st.info(f"Confidence: {np.max(proba)*100:.2f}%")

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
    st.info("Note: Deep learning models perform better with longer article text.")
