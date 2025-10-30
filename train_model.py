#What this script does:
# 1) Loads a CSV of short texts labeled as biased (1) or not (0)
# 2) Splits the data into train/test (stratified, so class ratios match)
# 3) Turns text into TF-IDF features (unigrams + bigrams)
# 4) Trains a Logistic Regression classifier
# 5) Evaluates on the test set and exports a small JSON model

# Notes on what I learned:
# - pandas for loading/cleaning CSVs (Kaggle + pandas docs)
# - scikit-learn for TF-IDF and logistic regression (sklearn docs)
# - stratify split to keep label proportions (sklearn guide)
# - exporting numpy types to JSON requires casting to built-ins
#   (Stack Overflow thread about np.float64 and json)


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import json

# 1) Load data 
# Expecting columns:
#   - "text": the sentence/paragraph
#   - "bias_present": 1 if biased, 0 if neutral
df = pd.read_csv("bias_dataset.csv")

# Make sure text is actually strings (handles NaN or numbers)
df["text"] = df["text"].astype(str)

# Make sure labels are ints (0/1). This avoids weird types later.
df["bias_present"] = df["bias_present"].astype(int)

# Separate features (X) and labels (y)
X = df["text"]
y = df["bias_present"]

# 2) Train/test split 
# test_size=0.2, 80/20 split
# random_state=42, reproducible results (so I can compare runs)
# stratify=y, keeps the class balance similar in train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3) Vectorizer + model 
# TF-IDF turns text into numbers:
#  - lowercase=True, normalize capitalization
#  - ngram_range=(1,2), unigrams + bigrams (helps catch short phrases like "female engineer")
#  - min_df=1, keep any term that appears at least once (dataset is small)
#  - max_features=4000 , caps vocabulary size to keep model tiny and fast
vec = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=1,
    max_features=4000
)

# Learn vocabulary + IDF on the training text only
X_train_tfidf = vec.fit_transform(X_train)
# Shape check (rows = examples, cols = features)
# print(X_train_tfidf.shape)

# LogisticRegression for a simple linear classifier:
#  - max_iter=2000 to make sure it converges
#  - class_weight="balanced" helps when labels are imbalanced
clf = LogisticRegression(max_iter=2000, class_weight="balanced")

# Fit the model on training features/labels
clf.fit(X_train_tfidf, y_train)

# 4) Evaluate 
# Transform test text using the *same* vectorizer (no .fit here)
X_test_tfidf = vec.transform(X_test)

# Make predictions on test set
y_pred = clf.predict(X_test_tfidf)

# Print a quick report: precision, recall, f1, support
print(classification_report(y_test, y_pred))

# 5) Safe export to JSON 
# The browser app needs:
#  - vocab: dict token to column index
#  - idf: list of IDF values (aligned with columns)
#  - coef: list of logistic regression weights (same order)
#  - intercept: single float
#  - ngram_range/use_idf so the UI knows what assumptions we used

# vec.vocabulary_ can have numpy/int types; cast them to built-ins.
vocab_py = {str(k): int(v) for k, v in vec.vocabulary_.items()}

# idf and coef are numpy arrays; convert to plain Python lists of floats.
# Important: order must match the column order the vectorizer uses.
idf_list = vec.idf_.astype(float).tolist()
coef_list = clf.coef_[0].astype(float).tolist()
intercept_val = float(clf.intercept_[0])

# Sanity check: lengths should match the number of features (columns)
# If these lengths disagree, something is off with the vectorizer or model.
# print(len(vocab_py), len(idf_list), len(coef_list))

model = {
    "vocab": vocab_py,                 # token --> column index
    "idf": idf_list,                   # list length = n_features
    "coef": coef_list,                 # list length = n_features (aligns with idf)
    "intercept": intercept_val,        # single number
    "ngram_range": list(vec.ngram_range),
    "use_idf": True
}

# Write JSON as UTF-8, ensure_ascii=False so tokens stay readable
with open("bias_model.json", "w", encoding="utf-8") as f:
    json.dump(model, f, ensure_ascii=False)

print("Model exported to bias_model.json")


