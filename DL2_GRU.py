# ==========================================
# FAST & OPTIMIZED GRU MODEL (CPU FRIENDLY)
# ==========================================

import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split

# ------------------------------------------
# 1. LOAD DATA
# ------------------------------------------
df = pd.read_csv("data/clean_train.csv")

X = df["clean_text"].astype(str).values
y = df["label"].values - 1

# ------------------------------------------
# 2. SPLIT
# ------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------
# 3. TOKENIZE
# ------------------------------------------
MAX_WORDS = 20000
MAX_LEN = 150   # 🔥 REDUCED

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

def preprocess(texts):
    return pad_sequences(
        tokenizer.texts_to_sequences(texts),
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

X_train_pad = preprocess(X_train)
X_val_pad   = preprocess(X_val)

with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# ------------------------------------------
# 4. FAST GRU MODEL
# ------------------------------------------
model = Sequential([
    Embedding(MAX_WORDS, 128, input_length=MAX_LEN),

    Bidirectional(GRU(64)),   # 🔥 SINGLE GRU, NO recurrent_dropout

    Dense(64, activation="relu"),
    Dropout(0.3),

    Dense(4, activation="softmax")
])

model.compile(
    optimizer=Adam(0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ------------------------------------------
# 5. TRAIN
# ------------------------------------------
history = model.fit(
    X_train_pad,
    y_train,
    validation_data=(X_val_pad, y_val),
    epochs=10,
    batch_size=128,   # 🔥 BIGGER BATCH
    callbacks=[EarlyStopping(patience=2, restore_best_weights=True)]
)

# ------------------------------------------
# 6. SAVE
# ------------------------------------------
model.save("models/gru_text_classifier.h5")

print("✅ FAST GRU TRAINED SUCCESSFULLY")
