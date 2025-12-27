# ==========================================
# PRETRAINED TRANSFORMER MODEL (DISTILBERT - FAST CPU)
# ==========================================

import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

from datasets import Dataset

# ------------------------------------------
# 1. LOAD DATA
# ------------------------------------------
df = pd.read_csv("data/clean_train.csv")

df["label"] = df["label"] - 1  # 1–4 → 0–3

X = df["clean_text"].astype(str)
y = df["label"]

# ------------------------------------------
# 2. TRAIN–VALIDATION SPLIT
# ------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

train_df = pd.DataFrame({"text": X_train, "label": y_train})
val_df   = pd.DataFrame({"text": X_val, "label": y_val})

# ------------------------------------------
# 3. HF DATASET
# ------------------------------------------
train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
val_dataset   = Dataset.from_pandas(val_df, preserve_index=False)

# ------------------------------------------
# 4. TOKENIZER
# ------------------------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)

def tokenize_function(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128   # 🔥 reduced for speed
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset   = val_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)
val_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)

# ------------------------------------------
# 5. MODEL
# ------------------------------------------
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=4
)

# ------------------------------------------
# 6. METRICS
# ------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro")
    }

# ------------------------------------------
# 7. TRAINING ARGUMENTS (CPU SAFE)
# ------------------------------------------
training_args = TrainingArguments(
    output_dir="./bert_results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_steps=500,
    save_total_limit=1,
    report_to="none"
)


# ------------------------------------------
# 8. TRAINER
# ------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# ------------------------------------------
# 9. TRAIN
# ------------------------------------------
trainer.train()

# ------------------------------------------
# 10. FINAL EVALUATION
# ------------------------------------------
results = trainer.evaluate()
print("\nEvaluation Results:", results)

# ------------------------------------------
# 11. SAVE MODEL
# ------------------------------------------
model.save_pretrained("models/distilbert_text_classifier")
tokenizer.save_pretrained("models/distilbert_text_classifier")

print("\n✅ DISTILBERT TRAINED & SAVED (CPU OPTIMIZED)")
