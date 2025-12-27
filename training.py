import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# -------------------- LOAD DATA --------------------
train_df = pd.read_csv("data/clean_train.csv")
test_df  = pd.read_csv("data/clean_test.csv")

# -------------------- MAP LABELS --------------------
label_map = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Technology"
}

train_df["label"] = train_df["label"].map(label_map)
test_df["label"]  = test_df["label"].map(label_map)

# -------------------- FEATURES & TARGET --------------------
X = train_df["clean_text"]
y = train_df["label"]

# -------------------- TRAIN TEST SPLIT --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------- LABEL ENCODING --------------------
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded  = label_encoder.transform(y_test)

# -------------------- SAVE LABEL CLASSES (SAFE) --------------------
os.makedirs("models", exist_ok=True)

# Convert to string dtype explicitly to avoid object arrays
classes = np.array(label_encoder.classes_, dtype=str)

# Save without pickle issues
np.save("models/label_classes.npy", classes)

print("✅ Label classes saved successfully")
print("Classes:", classes)
