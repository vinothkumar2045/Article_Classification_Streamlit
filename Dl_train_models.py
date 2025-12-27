import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

# =============================
# 1. LOAD DATA
# =============================
train_df = pd.read_csv("data/clean_train.csv")
test_df  = pd.read_csv("data/clean_test.csv")

X = train_df["clean_text"].astype(str).values
y = train_df["label"].values      # labels: 1,2,3,4
X_test = test_df["clean_text"].astype(str).values

print("\nClass distribution:")
print(pd.Series(y).value_counts())

# =============================
# 2. TRAIN / VALIDATION SPLIT
# =============================
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Convert labels to 0–3
y_train_enc = y_train - 1
y_val_enc   = y_val - 1

# =============================
# 3. TOKENIZATION & PADDING
# =============================
MAX_WORDS = 20000
MAX_LEN = 200

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

def pad_text(texts):
    seq = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")

X_train_pad = pad_text(X_train)
X_val_pad   = pad_text(X_val)
X_test_pad  = pad_text(X_test)

# =============================
# 4. CLASS WEIGHTS (VERY IMPORTANT)
# =============================
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train_enc),
    y=y_train_enc
)

class_weight_dict = dict(enumerate(class_weights))
print("\nClass Weights:", class_weight_dict)

# =============================
# 5. BUILD STRONGER MODEL
# =============================
model = Sequential([
    Embedding(MAX_WORDS, 128, input_length=MAX_LEN),

    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.4),

    Bidirectional(LSTM(64)),
    Dropout(0.4),

    Dense(64, activation="relu"),
    Dropout(0.3),

    Dense(4, activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =============================
# 6. TRAIN MODEL
# =============================
history = model.fit(
    X_train_pad,
    y_train_enc,
    validation_data=(X_val_pad, y_val_enc),
    epochs=15,
    batch_size=32,
    class_weight=class_weight_dict,
    verbose=1
)

# =============================
# 7. VALIDATION EVALUATION
# =============================
y_val_pred = np.argmax(model.predict(X_val_pad), axis=1)

print("\nValidation Classification Report:")
print(classification_report(
    y_val_enc,
    y_val_pred,
    target_names=["World", "Sports", "Business", "Technology"]
))

# =============================
# 8. SAVE MODEL
# =============================
model.save("models/lstm_text_classifier.h5")
print("\nModel saved successfully.")

# =============================
# 9. TEST PREDICTIONS
# =============================
y_test_pred = np.argmax(model.predict(X_test_pad), axis=1) + 1

label_map = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Technology"
}

test_df["predicted_category"] = [label_map[i] for i in y_test_pred]

test_df.to_csv("data/test_predictions.csv", index=False)

print("\nSample Test Predictions:")
print(test_df[["clean_text", "predicted_category"]].head())
