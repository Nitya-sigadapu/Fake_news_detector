#!/usr/bin/env python3
# train_model.py
"""
Safe training script:
 - robust CSV reader (tries multiple encodings + optional detector + fallback)
 - training logic (TF-IDF + Logistic Regression with GridSearch + MultinomialNB)
 - computes validation-based threshold and saves models + threshold.json
 - training only runs when executed directly (not on import)
"""

import os
import re
import sys
import json
import joblib
import traceback
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve

# ---------- CONFIG ----------
DATA_PATH = "data/fake_or_real_news.csv"   # update if needed (or keep sample.csv for testing)
MODELS_DIR = "models"
VEC_OUT = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
LR_OUT = os.path.join(MODELS_DIR, "logistic_regression.joblib")
NB_OUT = os.path.join(MODELS_DIR, "naive_bayes.joblib")
THRESH_OUT = os.path.join(MODELS_DIR, "threshold.json")
RANDOM_STATE = 42

os.makedirs(MODELS_DIR, exist_ok=True)

# ---------- robust CSV loader ----------
def robust_read_csv(path, encodings=None, **kwargs):
    """
    Try a list of encodings to read the CSV and return a DataFrame.
    Falls back to a 'replace' decoding strategy if all encodings fail.
    """
    encodings = encodings or ["utf-8", "utf-8-sig", "cp1252", "latin1", "iso-8859-1"]
    last_err = None
    for enc in encodings:
        try:
            # python engine can be more tolerant for weird CSV formatting
            df = pd.read_csv(path, encoding=enc, engine="python", **kwargs)
            print(f"[train_model] read CSV using encoding: {enc}")
            return df
        except Exception as e:
            last_err = e
            print(f"[train_model] encoding {enc} failed: {e!r}")

    # Try to detect encoding via charset_normalizer if installed (optional)
    try:
        from charset_normalizer import from_path
        result = from_path(path).best()
        if result:
            detected_enc = result.encoding
            try:
                df = pd.read_csv(path, encoding=detected_enc, engine="python", **kwargs)
                print(f"[train_model] read CSV using detected encoding: {detected_enc}")
                return df
            except Exception as e:
                last_err = e
                print(f"[train_model] detected encoding {detected_enc} failed: {e!r}")
    except Exception:
        # not installed or detection failed — continue to final fallback
        pass

    # Final fallback: read raw bytes and decode replacing invalid chars so file loads
    try:
        with open(path, "rb") as f:
            raw = f.read()
        text = raw.decode("utf-8", errors="replace")
        df = pd.read_csv(pd.io.common.StringIO(text), **kwargs)
        print("[train_model] read CSV by decoding with replacement of invalid bytes (utf-8 errors=replace)")
        return df
    except Exception as e:
        last_err = e
        print(f"[train_model] final fallback failed: {e!r}")

    # nothing worked — raise a clear error
    raise RuntimeError(f"Could not read CSV {path}. Last error: {last_err!r}")

# ---------- simple cleaner ----------
def simple_clean(text):
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = re.sub(r"http\S+", " ", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# ---------- training pipeline (function) ----------
def train_and_save(data_path=DATA_PATH):
    print("Loading data from:", data_path)
    df = robust_read_csv(data_path)

    # basic column checks
    if "label" not in df.columns or "text" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns.")

    # safer label mapping: handle common label variants
    if df["label"].dtype == object or df["label"].dtype == "O":
        df["label"] = (
            df["label"].astype(str)
            .str.strip()
            .str.lower()
            .replace({
                "real": 0, "truth": 0, "legit": 0, "true": 0,
                "fake": 1, "false": 1, "fraud": 1,
                "1": 1, "0": 0
            })
        )

    # if mapping produced NaNs, try fallback inference
    if df["label"].isna().any():
        def infer_label(x):
            s = str(x).lower()
            if "fake" in s or "false" in s:
                return 1
            if "real" in s or "true" in s:
                return 0
            try:
                return int(s)
            except Exception:
                return np.nan
        df["label"] = df["label"].fillna(df["label"].apply(infer_label))

    # final check
    if df["label"].isna().any():
        sample_bad = df.loc[df['label'].isna(), 'label'].unique()[:10]
        raise ValueError("Could not map some labels to 0/1. Sample problematic labels: " + str(sample_bad))

    # ensure numeric labels
    df["label"] = df["label"].astype(int)

    df["text"] = df["text"].fillna("").astype(str)
    df["clean_text"] = df["text"].apply(simple_clean)

    X = df["clean_text"].values
    y = df["label"].values

    # train/val/test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )
    print("Sizes — train/val/test:", len(X_train), len(X_val), len(X_test))


    # vectorizer (word unigrams + bigrams)
    vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1,2),
        analyzer="word",
        min_df=2,
        max_df=0.95,
        stop_words="english"
    )
    print("Fitting TF-IDF vectorizer...")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)

    # Logistic Regression (grid search for C)
    print("Training Logistic Regression with GridSearchCV...")
    param_grid = {"C": [0.01, 0.1, 0.5, 1.0, 5.0]}
    lr = LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
    gs = GridSearchCV(lr, param_grid, cv=3, scoring="f1", n_jobs=-1, verbose=1)
    gs.fit(X_train_vec, y_train)
    best_lr = gs.best_estimator_
    print("Best LR params:", gs.best_params_)

    # Validation probs -> choose threshold for FAKE (label=1)
    val_probs = best_lr.predict_proba(X_val_vec)[:, 1]
    prec, rec, thr = precision_recall_curve(y_val, val_probs)
    # thr has length = len(prec) - 1 ; compute F1 for thresholds only
    if len(thr) > 0:
        f1_for_thr = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-12)
        best_idx = int(np.nanargmax(f1_for_thr))
        best_threshold = float(thr[best_idx])
    else:
        best_threshold = 0.5
    print("Best threshold (val) for FAKE by F1:", best_threshold)

    # Evaluate on test set (LR)
    test_probs = best_lr.predict_proba(X_test_vec)[:, 1]
    y_test_pred = (test_probs >= best_threshold).astype(int)
    print("Test classification report (LR):")
    print(classification_report(y_test, y_test_pred))
    try:
        print("Test ROC AUC (LR):", roc_auc_score(y_test, test_probs))
    except Exception as e:
        print("[train_model] ROC AUC failed:", e)

    # Train MultinomialNB
    print("Training MultinomialNB...")
    nb = MultinomialNB()
    nb.fit(X_train_vec, y_train)
    nb_test_probs = nb.predict_proba(X_test_vec)[:, 1]
    y_test_pred_nb = (nb_test_probs >= best_threshold).astype(int)
    print("Test classification report (NB):")
    print(classification_report(y_test, y_test_pred_nb))
    try:
        print("Test ROC AUC (NB):", roc_auc_score(y_test, nb_test_probs))
    except Exception as e:
        print("[train_model] NB ROC AUC failed:", e)

    # Save vectorizer and models (with compression)
    print("Saving vectorizer and models...")
    try:
        joblib.dump(vectorizer, VEC_OUT, compress=3)
        joblib.dump(best_lr, LR_OUT, compress=3)
        joblib.dump(nb, NB_OUT, compress=3)
        with open(THRESH_OUT, "w") as f:
            json.dump({"prob_fake_threshold": best_threshold}, f, indent=2)
    except Exception as e:
        print("[train_model] ERROR saving models:", e)
        traceback.print_exc()
        raise

    print("Saved:", VEC_OUT, LR_OUT, NB_OUT, THRESH_OUT)
    return True

# ---------- only run training when the script is executed directly ----------
if __name__ == "__main__":
    dp = DATA_PATH
    if len(sys.argv) > 1:
        dp = sys.argv[1]
    try:
        train_and_save(dp)
    except Exception as e:
        print("[train_model] ERROR:", e)
        traceback.print_exc()
        raise

