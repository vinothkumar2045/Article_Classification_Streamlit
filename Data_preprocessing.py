import pandas as pd
import re

# =============================
# 1. LOAD DATA
# =============================
train_df = pd.read_csv("data/clean_train.csv")
test_df = pd.read_csv("data/clean_test.csv")

# =============================
# 2. TEXT CLEANING FUNCTION
# =============================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# =============================
# 3. PREPROCESS
# =============================
for df in [train_df, test_df]:
    df['text'] = df['title'] + " " + df['description']
    df['clean_text'] = df['text'].apply(clean_text)

# =============================
# 4. VERIFY
# =============================
print(train_df.head())
print(test_df.head())
