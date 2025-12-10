import os
import re
import sys
import json
import joblib
import traceback
import pandas as pd
import numpy as np
from datetime import datetime

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
np.random.seed(RANDOM_STATE)

# ---------- robust CSV loader ----------
def robust_read_csv(path, encodings=None, **kwargs):
    encodings = encodings or ["utf-8", "utf-8-sig", "cp1252", "latin1", "iso-8859-1"]
    last_err = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, engine="python", **kwargs)
            print(f"[train_model] read CSV using encoding: {enc}")
            return df
        except Exception as e:
            last_err = e
            print(f"[train_model] encoding {enc} failed: {e!r}")

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
        pass

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

# ---------- training pipeline ----------
def train_and_save(data_path=DATA_PATH, save_timestamp=False):
    print("Loading data from:", data_path)
    df = robust_read_csv(data_path)

    if "label" not in df.columns or "text" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns.")

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

    if df["label"].isna().any():
        sample_bad = df.loc[df['label'].isna(), 'label'].unique()[:10]
        raise ValueError("Could not map some labels to 0/1. Sample problematic labels: " + str(sample_bad))

    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].fillna("").astype(str)
    df["clean_text"] = df["text"].apply(simple_clean)

    X = df["clean_text"].values
    y = df["label"].values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )
    print("Sizes — train/val/test:", len(X_train), len(X_val), len(X_test))

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

    print("Training Logistic Regression with GridSearchCV...")
    param_grid = {"C": [0.01, 0.1, 0.5, 1.0, 5.0]}
    lr = LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
    gs = GridSearchCV(lr, param_grid, cv=3, scoring="f1", n_jobs=-1, verbose=1)
    gs.fit(X_train_vec, y_train)
    best_lr = gs.best_estimator_
    print("Best LR params:", gs.best_params_)

    val_probs = best_lr.predict_proba(X_val_vec)[:, 1]
    prec, rec, thr = precision_recall_curve(y_val, val_probs)
    if len(thr) > 0:
        f1_for_thr = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-12)
        best_idx = int(np.nanargmax(f1_for_thr))
        best_threshold = float(thr[best_idx])
    else:
        best_threshold = 0.5
    print("Best threshold (val) for FAKE by F1:", best_threshold)

    test_probs = best_lr.predict_proba(X_test_vec)[:, 1]
    y_test_pred = (test_probs >= best_threshold).astype(int)
    print("Test classification report (LR):")
    print(classification_report(y_test, y_test_pred))

    try:
        if len(np.unique(y_test)) > 1:
            print("Test ROC AUC (LR):", roc_auc_score(y_test, test_probs))
        else:
            print("[train_model] ROC AUC skipped: only one class present in test labels.")
    except Exception as e:
        print("[train_model] ROC AUC failed:", e)

    print("Training MultinomialNB...")
    nb = MultinomialNB()
    nb.fit(X_train_vec, y_train)
    nb_test_probs = nb.predict_proba(X_test_vec)[:, 1]
    y_test_pred_nb = (nb_test_probs >= best_threshold).astype(int)
    print("Test classification report (NB):")
    print(classification_report(y_test, y_test_pred_nb))

    try:
        if len(np.unique(y_test)) > 1:
            print("Test ROC AUC (NB):", roc_auc_score(y_test, nb_test_probs))
        else:
            print("[train_model] NB ROC AUC skipped: only one class present in test labels.")
    except Exception as e:
        print("[train_model] NB ROC AUC failed:", e)

    # optional timestamping
    if save_timestamp:
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        vec_out = VEC_OUT.replace(".joblib", f".{ts}.joblib")
        lr_out = LR_OUT.replace(".joblib", f".{ts}.joblib")
        nb_out = NB_OUT.replace(".joblib", f".{ts}.joblib")
        thresh_out = THRESH_OUT.replace(".json", f".{ts}.json")
    else:
        vec_out, lr_out, nb_out, thresh_out = VEC_OUT, LR_OUT, NB_OUT, THRESH_OUT

    print("Saving vectorizer and models...")
    try:
        joblib.dump(vectorizer, vec_out, compress=3)
        joblib.dump(best_lr, lr_out, compress=3)
        joblib.dump(nb, nb_out, compress=3)
        with open(thresh_out, "w") as f:
            json.dump({"prob_fake_threshold": best_threshold}, f, indent=2)
    except Exception as e:
        print("[train_model] ERROR saving models:", e)
        traceback.print_exc()
        raise

    print("Saved:", vec_out, lr_out, nb_out, thresh_out)
    return {
        "vec": vec_out,
        "lr": lr_out,
        "nb": nb_out,
        "threshold": best_threshold
    }


if __name__ == "__main__":
    dp = DATA_PATH
    save_ts = False
    if len(sys.argv) > 1:
        dp = sys.argv[1]
    if "--save-ts" in sys.argv:
        save_ts = True
    try:
        result = train_and_save(dp, save_timestamp=save_ts)
        print("Training finished. Summary:", result)
    except Exception as e:
        print("[train_model] ERROR:", e)
        traceback.print_exc()
        raise



          
