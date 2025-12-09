# app.py
"""
Fake News Detector — Safe & consistent
- Uses prob_fake threshold (from models/threshold.json if present)
- Rule-based overrides (death / impossible science / policy hoaxes)
- Safe model loading + graceful warnings
- Model selector (LogReg / Naive Bayes), SHAP optional, CSV bulk
"""

import streamlit as st
import joblib
import os
import json
import re
import numpy as np
import pandas as pd
from io import BytesIO

# Optional SHAP (do not require it)
try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

st.set_page_config(page_title="Fake News Detector", layout="wide")

MODELS_DIR = "models"
VECT_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
LR_PATH = os.path.join(MODELS_DIR, "logistic_regression.joblib")
NB_PATH = os.path.join(MODELS_DIR, "naive_bayes.joblib")
THRESH_PATH = os.path.join(MODELS_DIR, "threshold.json")

# ---------- utilities ----------
def simple_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = re.sub(r"http\S+", " ", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def safe_load_joblib(path):
    try:
        return joblib.load(path)
    except Exception as e:
        # do not crash the app; return None and show warning later
        return None

def load_saved_threshold(path):
    try:
        with open(path, "r") as f:
            j = json.load(f)
            val = j.get("prob_fake_threshold")
            if val is None:
                return None
            return float(val)
    except Exception:
        return None

def df_to_csv_bytes(df: pd.DataFrame):
    out = BytesIO()
    df.to_csv(out, index=False)
    return out.getvalue()

# token importance helpers (kept from your code)
def top_tokens_for_instance(vectorizer, X_row, model, topn=10):
    feat_names = np.array(vectorizer.get_feature_names_out())
    row = X_row.toarray().ravel()
    if row.sum() == 0:
        return pd.DataFrame(columns=["token", "score"])
    if model is None:
        idx_sorted = row.argsort()[::-1][:topn]
        return pd.DataFrame({"token": feat_names[idx_sorted], "score": row[idx_sorted]})
    if hasattr(model, "coef_"):
        coef = model.coef_
        if coef.shape[0] > 1:
            coef = (coef[0] - coef[1]).ravel()
        else:
            coef = coef.ravel()
        contrib = row * coef
        idx_sorted = np.argsort(contrib)[::-1][:topn]
        return pd.DataFrame({"token": feat_names[idx_sorted], "score": contrib[idx_sorted]})
    if hasattr(model, "feature_log_prob_"):
        flp = model.feature_log_prob_
        if flp.shape[0] == 2:
            log_ratio = flp[1] - flp[0]
            contrib = row * log_ratio
            idx_sorted = np.argsort(contrib)[::-1][:topn]
            return pd.DataFrame({"token": feat_names[idx_sorted], "score": contrib[idx_sorted]})
    idx_sorted = row.argsort()[::-1][:topn]
    return pd.DataFrame({"token": feat_names[idx_sorted], "score": row[idx_sorted]})

# ---------- load models + threshold ----------
VECT = safe_load_joblib(VECT_PATH) if os.path.exists(VECT_PATH) else None
LR_MODEL = safe_load_joblib(LR_PATH) if os.path.exists(LR_PATH) else None
NB_MODEL = safe_load_joblib(NB_PATH) if os.path.exists(NB_PATH) else None
SAVED_THRESHOLD = load_saved_threshold(THRESH_PATH) if os.path.exists(THRESH_PATH) else None

# Inform user about loading status (do not crash)
missing = []
if VECT is None:
    missing.append("vectorizer (models/tfidf_vectorizer.joblib)")
if LR_MODEL is None:
    missing.append("logistic_regression (models/logistic_regression.joblib)")
if NB_MODEL is None:
    # NB optional
    pass

# ---------- Sidebar & options ----------
st.sidebar.header("Options")
model_choice = st.sidebar.selectbox("Choose model", ("Logistic Regression", "Naive Bayes"))
use_saved_threshold = st.sidebar.checkbox("Use saved threshold from training (prob_fake)", value=True)
default_thresh = SAVED_THRESHOLD if (use_saved_threshold and SAVED_THRESHOLD is not None) else 0.50
threshold = st.sidebar.slider("FAKE probability threshold (prob_fake >= threshold → FAKE)", 0.0, 1.0, float(default_thresh), 0.01)
top_tokens = st.sidebar.slider("Top tokens to show", 0, 20, 8)
show_shap = st.sidebar.checkbox("Use SHAP explanations (if available)", value=False)
enable_rules = st.sidebar.checkbox("Enable rule-based overrides (death / impossible science / policy hoaxes)", value=True)

st.title("📰 Fake News Detector")
st.markdown("TF-IDF + classical models (demo).")

if missing:
    st.warning("Model loading notes: " + "; ".join(missing) + ". App needs models/tfidf_vectorizer.joblib and logistic_regression.joblib to run.")

# ---------- helper: predict single ----------
def predict_single(model, vectorizer, text):
    cleaned = simple_clean(text)
    X = vectorizer.transform([cleaned])
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        # we expect sklearn columns aligned with labels [0,1] -> [REAL, FAKE]
        prob_real = float(probs[0])
        prob_fake = float(probs[1])
    else:
        pred = model.predict(X)[0]
        prob_fake = 1.0 if pred == 1 else 0.0
        prob_real = 1.0 - prob_fake
    return prob_real, prob_fake, X

# ---------- Input area ----------
st.subheader("Single prediction")
text = st.text_area("Paste a headline or article", height=150)
cols = st.columns([1,1,1])
with cols[0]:
    btn_predict = st.button("Predict")
with cols[1]:
    st.write("")
with cols[2]:
    if st.button("Load example"):
        text = "breaking: celebrity scandal shocks fans"

# model selection
model = LR_MODEL if model_choice == "Logistic Regression" else NB_MODEL

if VECT is None:
    st.error("Vectorizer missing — add models/tfidf_vectorizer.joblib to repository/models and redeploy.")
    st.stop()

if model is None:
    st.warning(f"Selected model ({model_choice}) not available. Please choose a different model or add the model file.")
    st.stop()

# ---------- rules for overrides ----------
def detect_rule_flags(s: str):
    s_l = s.lower()
    flags = []
    # death claims
    if re.search(r"\b(died|dead|passes away|was found dead)\b", s_l):
        flags.append("death_claim")
    # impossible science phrases (extend as needed)
    impossible_phrases = ["sun will shut down", "earth will stop rotating", "moon will crash", "end of the world", "sun to shut"]
    if any(p in s_l for p in impossible_phrases):
        flags.append("impossible_science")
    # policy hoaxes
    if re.search(r"\b(ban .* mobile|ban mobile phones|start charging users)\b", s_l):
        flags.append("policy_hoax")
    return flags

# ---------- Prediction logic ----------
if btn_predict:
    if not text or not text.strip():
        st.warning("Please enter text to predict.")
    else:
        flags = detect_rule_flags(text) if enable_rules else []
        prob_real, prob_fake, X_row = predict_single(model, VECT, text)

        overridden = False
        override_reason = None
        if enable_rules and flags:
            # simple policy: if any strong rule hit, mark FAKE (needs verification)
            overridden = True
            override_reason = ", ".join(flags)
            final_label = "FAKE"
        else:
            final_label = "FAKE" if prob_fake >= threshold else "REAL"

        # show result
        st.markdown("### Prediction: " + (":green[REAL]" if final_label == "REAL" else ":red[FAKE]"))
        st.write(f"prob_real = {prob_real:.4f}    prob_fake = {prob_fake:.4f}")
        if SAVED_THRESHOLD is not None:
            st.caption(f"Saved prob_fake threshold from training: {SAVED_THRESHOLD:.4f}")
        if overridden:
            st.warning(f"Rule override applied ({override_reason}) — flagged as FAKE for manual verification.")

        # probability bar
        prob_df = pd.DataFrame({"class":["REAL","FAKE"], "prob":[prob_real, prob_fake]}).set_index("class")
        st.bar_chart(prob_df)

        # tokens / contributions
        st.subheader("Key tokens (model-weighted)")
        tok_df = top_tokens_for_instance(VECT, X_row, model, topn=top_tokens)
        if tok_df.empty:
            st.info("No tokens found after vectorization.")
        else:
            tok_df["abs_score"] = np.abs(tok_df["score"])
            tok_df = tok_df.sort_values("abs_score", ascending=False).reset_index(drop=True)
            tok_df_display = tok_df.copy()
            tok_df_display["score"] = tok_df_display["score"].round(6)
            st.dataframe(tok_df_display[["token","score"]], height=260)
            st.bar_chart(tok_df_display.head(10).set_index("token")["score"])

        # SHAP (optional)
        st.subheader("Explanation")
        if show_shap and _HAS_SHAP:
            try:
                explainer = shap.LinearExplainer(model, VECT)
                shap_vals = explainer.shap_values(X_row)
                fv = pd.DataFrame({
                    "feature": VECT.get_feature_names_out(),
                    "value": X_row.toarray().ravel(),
                    "shap": shap_vals.ravel() if hasattr(shap_vals, "ravel") else shap_vals[0].ravel()
                })
                fv = fv[fv["value"] != 0].sort_values("shap", key=lambda x: np.abs(x), ascending=False).head(top_tokens)
                st.table(fv[["feature","value","shap"]])
            except Exception as e:
                st.info("SHAP failed; showing coefficient-based contributions above.")
        else:
            if show_shap and not _HAS_SHAP:
                st.info("SHAP not installed in the environment; install shap to enable.")

# ---------- Batch CSV ----------
st.markdown("---")
st.subheader("Bulk predictions (CSV)")

uploaded = st.file_uploader("Upload a CSV with a column named 'text'", type=["csv"])
if uploaded:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error("Could not read CSV: " + str(e))
        df = None

    if df is not None:
        if "text" not in df.columns:
            st.error("CSV must contain a 'text' column. Example: text,label")
        else:
            st.write(f"Dataset rows: {len(df)}")
            texts = df["text"].fillna("").astype(str).tolist()
            prob_reals = []
            prob_fakes = []
            pred_labels = []
            flags_list = []
            for t in texts:
                flags = detect_rule_flags(t) if enable_rules else []
                pr, pf, _ = predict_single(model, VECT, t)
                if enable_rules and flags:
                    lab = "FAKE"
                else:
                    lab = "FAKE" if pf >= threshold else "REAL"
                prob_reals.append(pr); prob_fakes.append(pf); pred_labels.append(lab); flags_list.append(",".join(flags) if flags else "")
            df_out = df.copy()
            df_out["prediction"] = pred_labels
            df_out["prob_real"] = prob_reals
            df_out["prob_fake"] = prob_fakes
            df_out["rule_flags"] = flags_list
            st.dataframe(df_out.head(50))
            st.download_button("Download results CSV", data=df_to_csv_bytes(df_out), file_name="predictions.csv", mime="text/csv")

st.markdown("---")
st.caption("Demo: TF-IDF + classical models. Uses prob_fake threshold saved in models/threshold.json when available.")

