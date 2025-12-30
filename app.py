# ==========================================
# STREAMLIT MULTI-MODEL ARTICLE CLASSIFICATION
# WITH RDS AUTHENTICATION
# ==========================================

import streamlit as st
import numpy as np
import torch
import pickle
import pandas as pd
import hashlib

from joblib import load
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sqlalchemy import create_engine, text

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(page_title="Article Classification System", layout="centered")
st.title("📰 Multi-Model Article Classification System")

# =============================
# DATABASE CONFIG (CHANGE PASSWORD)
# =============================
DB_HOST = "streamlit-db.c10c0s8c6mju.ap-south-1.rds.amazonaws.com"
DB_PORT = "5432"
DB_NAME = "articledb"
DB_USER = "dbadmin"
DB_PASSWORD = "YOUR_DB_PASSWORD"   # 👈 change this

@st.cache_resource
def get_db_engine():
    return create_engine(
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

# =============================
# AUTH HELPERS
# =============================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    engine = get_db_engine()
    query = text("""
        INSERT INTO users (username, password)
        VALUES (:username, :password)
    """)
    with engine.connect() as conn:
        conn.execute(query, {
            "username": username,
            "password": hash_password(password)
        })
        conn.commit()

def validate_login(username, password):
    engine = get_db_engine()
    query = text("""
        SELECT * FROM users
        WHERE username = :username AND password = :password
    """)
    with engine.connect() as conn:
        result = conn.execute(query, {
            "username": username,
            "password": hash_password(password)
        }).fetchone()
    return result is not None

def log_to_rds(username, model_used):
    engine = get_db_engine()
    query = text("""
        INSERT INTO user_logins (username, model_used)
        VALUES (:username, :model_used)
    """)
    with engine.connect() as conn:
        conn.execute(query, {
            "username": username,
            "model_used": model_used
        })
        conn.commit()

# =============================
# SESSION STATE
# =============================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "page" not in st.session_state:
    st.session_state.page = "🔐 Login"

# =============================
# SIDEBAR NAVIGATION
# =============================
pages = ["🔐 Login", "🧠 Article Classification", "📊 Model Comparison"]
page = st.sidebar.radio("Navigation", pages, index=pages.index(st.session_state.page))
st.session_state.page = page

# =============================
# LABEL MAP
# =============================
label_map = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Technology"
}

# =============================
# LOAD MODELS (LOCAL PATH)
# =============================
@st.cache_resource
def load_ml_model():
    return load("models/best_model_LogisticRegression.joblib")

@st.cache_resource
def load_lstm_model():
    return load_model("models/lstm_text_classifier.h5")

@st.cache_resource
def load_gru_model():
    return load_model("models/gru_text_classifier.h5")

@st.cache_resource
def load_tokenizer():
    with open("models/tokenizer.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_transformer():
    tokenizer = DistilBertTokenizerFast.from_pretrained("models/distilbert_text_classifier")
    model = DistilBertForSequenceClassification.from_pretrained("models/distilbert_text_classifier")
    model.eval()
    return tokenizer, model

# ======================================================
# PAGE 1: LOGIN / REGISTER
# ======================================================
if page == "🔐 Login":
    st.subheader("🔐 User Authentication")

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if validate_login(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                log_to_rds(username, "LOGIN")
                st.session_state.page = "🧠 Article Classification"
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid username or password")

    with tab2:
        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")

        if st.button("Register"):
            try:
                register_user(new_user, new_pass)
                st.success("User registered successfully. Please login.")
            except:
                st.error("Username already exists")

# ======================================================
# PAGE 2: ARTICLE CLASSIFICATION
# ======================================================
elif page == "🧠 Article Classification":
    if not st.session_state.logged_in:
        st.warning("Please login first")
    else:
        st.success(f"Logged in as: {st.session_state.username}")

        article = st.text_area("Enter Article Text", height=200)
        model_type = st.selectbox(
            "Select Model",
            [
                "Machine Learning (Logistic Regression)",
                "Deep Learning (LSTM)",
                "Deep Learning (GRU)",
                "Transformer (DistilBERT)"
            ]
        )

        if st.button("Predict"):
            if model_type == "Machine Learning (Logistic Regression)":
                proba = load_ml_model().predict_proba([article])[0]

            elif model_type == "Deep Learning (LSTM)":
                seq = pad_sequences(load_tokenizer().texts_to_sequences([article]), maxlen=200)
                proba = load_lstm_model().predict(seq)[0]

            elif model_type == "Deep Learning (GRU)":
                seq = pad_sequences(load_tokenizer().texts_to_sequences([article]), maxlen=150)
                proba = load_gru_model().predict(seq)[0]

            else:
                tokenizer, model = load_transformer()
                inputs = tokenizer(article, return_tensors="pt", truncation=True, padding=True)
                with torch.no_grad():
                    proba = torch.softmax(model(**inputs).logits, dim=1).numpy()[0]

            pred_idx = int(np.argmax(proba)) + 1
            st.success(f"Predicted Category: {label_map[pred_idx]}")
            st.info(f"Confidence: {np.max(proba)*100:.2f}%")

            log_to_rds(st.session_state.username, model_type)

# ======================================================
# PAGE 3: MODEL COMPARISON
# ======================================================
elif page == "📊 Model Comparison":
    df = pd.DataFrame({
        "Model": ["Logistic Regression", "LSTM", "GRU", "DistilBERT"],
        "Accuracy (%)": [90.6, 92.8, 94.1, 98.2],
        "F1 Score": [0.90, 0.92, 0.94, 0.98]
    })
    st.dataframe(df, use_container_width=True)
