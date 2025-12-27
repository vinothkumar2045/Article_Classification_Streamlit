import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
from joblib import dump, load

# =============================
# 0. CREATE REQUIRED FOLDERS
# =============================
os.makedirs("models", exist_ok=True)

# =============================
# 1. LOAD CLEANED DATA
# =============================
train_df = pd.read_csv("data/clean_train.csv")
test_df  = pd.read_csv("data/clean_test.csv")

label_map = {1: "World", 2: "Sports", 3: "Business", 4: "Technology"}

X = train_df["clean_text"]
y = train_df["label"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =============================
# 2. DEFINE OPTIMIZED MODELS
# =============================
models = {

    "MultinomialNB": Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2
        )),
        ("clf", MultinomialNB(alpha=0.5))
    ]),

    "LogisticRegression": Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=8000,          # 🔥 increased
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2
        )),
        ("clf", LogisticRegression(
            max_iter=1500,              # 🔥 more stable
            n_jobs=-1,
            C=2.0,                      # 🔥 tuned
            solver="lbfgs"
        ))
    ]),

    "SGDClassifier": Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=6000,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2
        )),
        ("clf", SGDClassifier(
            loss="log_loss",
            max_iter=1500,
            random_state=42
        ))
    ])
}

# =============================
# 3. TRAIN, EVALUATE & LOG
# =============================
mlflow.set_experiment("AG_News_Optimized_Text_Models")

best_acc = 0
best_model_name = None

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)

        acc = accuracy_score(y_val, y_val_pred)

        print(f"Validation Accuracy: {acc:.4f}")
        print(classification_report(
            y_val,
            y_val_pred,
            target_names=[label_map[i] for i in sorted(label_map)]
        ))

        mlflow.log_param("model", name)
        mlflow.log_metric("val_accuracy", acc)

        mlflow.sklearn.log_model(model, artifact_path=name)

        if acc > best_acc:
            best_acc = acc
            best_model_name = name
            dump(model, f"models/best_model_{name}.joblib")

print(f"\n✅ BEST MODEL: {best_model_name} | Accuracy: {best_acc:.4f}")

# =============================
# 4. TEST PREDICTIONS
# =============================
best_model = load(f"models/best_model_{best_model_name}.joblib")

y_test_pred = best_model.predict(test_df["clean_text"])
test_df["predicted_category"] = y_test_pred.map(label_map)

print("\nSample Test Predictions:")
print(test_df[["text", "predicted_category"]].head())
