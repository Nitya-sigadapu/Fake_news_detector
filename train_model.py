#!/usr/bin/env python3
# train_model.py

import os
import re
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# ---------- CONFIG ----------
DATA_PATH = "data/sample.csv"  # change to your dataset path if you have one
TEXT_COL = "text"
LABEL_COL = "label"  # expects 0 (REAL) or 1 (FAKE); mapping handled below
MODELS_DIR = "models"
VEC_OUT = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
LR_OUT = os.path.join(MODELS_DIR, "logistic_regression.joblib")
RANDOM_STATE = 42

# ---------- SIMPLE CLEANER (no NLTK) ----------
def simple_clean(text):
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = re.sub(r"http\S+", " ", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# ---------- LOAD DATA ----------
print("Loading data from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)

# If labels are strings like 'REAL'/'FAKE', map to 0/1
if df[LABEL_COL].dtype == object:
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip().str.lower().map({"real": 0, "fake": 1})

df[TEXT_COL] = df[TEXT_COL].fillna("")
df["clean_text"] = df[TEXT_COL].apply(simple_clean)

X = df["clean_text"].values
y = df[LABEL_COL].values

# ---------- SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

# ---------- TF-IDF ----------
vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1,2),
    min_df=3,
    max_df=0.9,
    stop_words="english"
)
print("Fitting TF-IDF vectorizer...")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---------- TRAIN ----------
clf = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000, random_state=RANDOM_STATE)
print("Training Logistic Regression...")
clf.fit(X_train_vec, y_train)

# ---------- EVAL ----------
y_pred = clf.predict(X_test_vec)
y_prob = clf.predict_proba(X_test_vec)[:, 1]
print("Classification report:")
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

# ---------- SAVE ----------
os.makedirs(MODELS_DIR, exist_ok=True)
print("Saving vectorizer to:", VEC_OUT)
joblib.dump(vectorizer, VEC_OUT)
print("Saving classifier to:", LR_OUT)
joblib.dump(clf, LR_OUT)

print("Done. Saved models in", MODELS_DIR)
