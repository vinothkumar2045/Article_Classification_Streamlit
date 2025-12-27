# ================================
# STEP 3: EXPLORATORY DATA ANALYSIS
# ================================

# ----------------
# 1. IMPORT LIBRARIES
# ----------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

# ----------------
# 2. LOAD DATASET
# ----------------
df = pd.read_csv("data/clean_train.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# ----------------
# 3. LABEL MAPPING
# ----------------
label_map = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Technology"
}

df["category"] = df["label"].map(label_map)

# ----------------
# 4. CLASS DISTRIBUTION
# ----------------
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="category")
plt.title("Class Distribution of Articles")
plt.xlabel("Category")
plt.ylabel("Number of Articles")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# ----------------
# 5. WORD CLOUD PER CATEGORY
# ----------------
def generate_wordcloud(text, title):
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        max_words=200
    ).generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.show()

for cat in df["category"].unique():
    text = " ".join(df[df["category"] == cat]["clean_text"].astype(str))
    generate_wordcloud(text, f"Word Cloud - {cat}")

# ----------------
# 6. WORD COUNT ANALYSIS
# ----------------
df["word_count"] = df["clean_text"].apply(lambda x: len(str(x).split()))

# Average word count per category
avg_word_count = df.groupby("category")["word_count"].mean()

plt.figure(figsize=(8, 5))
avg_word_count.plot(kind="bar")
plt.title("Average Word Count per Category")
plt.xlabel("Category")
plt.ylabel("Average Word Count")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# ----------------
# 7. ARTICLE LENGTH DISTRIBUTION
# ----------------
plt.figure(figsize=(8, 5))
sns.histplot(df["word_count"], bins=50, kde=True)
plt.title("Article Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# ----------------
# 8. HEATMAP OF FREQUENT WORDS PER CLASS
# ----------------
vectorizer = CountVectorizer(
    stop_words="english",
    max_features=15
)

categories = df["category"].unique()
heatmap_data = []

for cat in categories:
    text = df[df["category"] == cat]["clean_text"].astype(str)
    X = vectorizer.fit_transform(text)
    word_freq = np.sum(X.toarray(), axis=0)
    heatmap_data.append(word_freq)

heatmap_df = pd.DataFrame(
    heatmap_data,
    columns=vectorizer.get_feature_names_out(),
    index=categories
)

plt.figure(figsize=(14, 6))
sns.heatmap(heatmap_df, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Frequent Words per Category Heatmap")
plt.xlabel("Words")
plt.ylabel("Category")
plt.tight_layout()
plt.show()

# ----------------
# 9. EDA SUMMARY OUTPUT
# ----------------
print("\nEDA Summary:")
print("-------------")
print("Number of Categories:", df['category'].nunique())
print("Average Word Count per Category:")
print(avg_word_count)
