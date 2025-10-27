import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import json

# 1) Load data
df = pd.read_csv("bias_dataset.csv")
df["text"] = df["text"].astype(str)
df["bias_present"] = df["bias_present"].astype(int)

X = df["text"]
y = df["bias_present"]

# 2) Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3) Vectorizer + model
vec = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=1,
    max_features=4000
)
X_train_tfidf = vec.fit_transform(X_train)

clf = LogisticRegression(max_iter=2000, class_weight="balanced")
clf.fit(X_train_tfidf, y_train)

# 4) Evaluate
X_test_tfidf = vec.transform(X_test)
y_pred = clf.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))

# 5) Safe export to JSON
vocab_py = {str(k): int(v) for k, v in vec.vocabulary_.items()}
model = {
    "vocab": vocab_py,
    "idf": vec.idf_.astype(float).tolist(),
    "coef": clf.coef_[0].astype(float).tolist(),
    "intercept": float(clf.intercept_[0]),
    "ngram_range": list(vec.ngram_range),
    "use_idf": True
}

with open("bias_model.json", "w", encoding="utf-8") as f:
    json.dump(model, f, ensure_ascii=False)

print("Model exported to bias_model.json")
