# ==========================================
# STREAMLIT MULTI-MODEL ARTICLE CLASSIFICATION APP
# ==========================================

import streamlit as st
import numpy as np
import torch
from joblib import load
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import pickle
import pandas as pd

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Article Classification System",
    layout="centered"
)

st.title("📰 Multi-Model Article Classification System")

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
st.sidebar.title("📌 Navigation")

pages = ["🔐 Login", "🧠 Article Classification", "📊 Model Comparison"]

page = st.sidebar.radio(
    "Go to",
    pages,
    index=pages.index(st.session_state.page)
)

# Keep sidebar + session in sync
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
# LOAD MODELS
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
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        "models/distilbert_text_classifier"
    )
    model = DistilBertForSequenceClassification.from_pretrained(
        "models/distilbert_text_classifier"
    )
    model.eval()
    return tokenizer, model

# ======================================================
# PAGE 1: LOGIN
# ======================================================
if page == "🔐 Login":
    st.subheader("🔐 User Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username.strip() == "" or password.strip() == "":
            st.warning("Please enter username and password")
        else:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.page = "🧠 Article Classification"
            st.success(f"Welcome {username} 👋")
            st.info("Redirecting to Article Classification...")
            st.rerun()

# ======================================================
# PAGE 2: ARTICLE CLASSIFICATION
# ======================================================
elif page == "🧠 Article Classification":
    st.subheader("🧠 Article Classification")

    if not st.session_state.logged_in:
        st.warning("⚠️ Please login first")
    else:
        st.success(f"Logged in as: **{st.session_state.username}**")

        article = st.text_area(
            "Enter Article Text",
            height=200,
            placeholder="Paste article content here..."
        )

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
            if article.strip() == "":
                st.warning("Please enter article text")
            else:
                # -------- PREDICTION --------
                if model_type == "Machine Learning (Logistic Regression)":
                    model = load_ml_model()
                    proba = model.predict_proba([article])[0]

                elif model_type == "Deep Learning (LSTM)":
                    model = load_lstm_model()
                    tokenizer = load_tokenizer()
                    seq = pad_sequences(
                        tokenizer.texts_to_sequences([article]),
                        maxlen=200,
                        padding="post"
                    )
                    proba = model.predict(seq, verbose=0)[0]

                elif model_type == "Deep Learning (GRU)":
                    model = load_gru_model()
                    tokenizer = load_tokenizer()
                    seq = pad_sequences(
                        tokenizer.texts_to_sequences([article]),
                        maxlen=150,
                        padding="post"
                    )
                    proba = model.predict(seq, verbose=0)[0]

                else:
                    tokenizer, model = load_transformer()
                    inputs = tokenizer(
                        article,
                        return_tensors="pt",
                        truncation=True,
                        padding=True,
                        max_length=128
                    )
                    with torch.no_grad():
                        outputs = model(**inputs)
                        proba = torch.softmax(
                            outputs.logits, dim=1
                        ).cpu().numpy()[0]

                pred_idx = int(np.argmax(proba)) + 1
                pred_label = label_map[pred_idx]
                confidence = np.max(proba) * 100

                st.success(f"✅ Predicted Category: **{pred_label}**")
                st.info(f"📊 Confidence: **{confidence:.2f}%**")

                st.subheader("Class Probabilities")
                for i, cls in label_map.items():
                    st.write(f"{cls}: {proba[i-1]*100:.2f}%")

                # =============================
                # FUTURE RDS LOGGING
                # =============================
                # log_to_rds(
                #     username=st.session_state.username,
                #     model_used=model_type
                # )

# ======================================================
# PAGE 3: MODEL COMPARISON
# ======================================================
elif page == "📊 Model Comparison":
    st.subheader("📊 Model Performance Comparison")

    df = pd.DataFrame({
        "Model Type": ["ML", "DL", "DL", "Transformer"],
        "Model Name": [
            "Logistic Regression",
            "LSTM",
            "GRU",
            "DistilBERT"
        ],
        "Accuracy (%)": [90.6, 92.8, 94.1, 98.2],
        "F1-Score": [0.90, 0.92, 0.94, 0.98],
        "Pros": [
            "Fast & simple",
            "Good sequence modeling",
            "Efficient & accurate",
            "Best contextual understanding"
        ],
        "Cons": [
            "Limited context",
            "Slow training",
            "Needs tuning",
            "High resource usage"
        ]
    })

    st.dataframe(df, use_container_width=True)
    st.success("🏆 DistilBERT selected as final production model")
