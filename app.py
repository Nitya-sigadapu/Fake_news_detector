# app.py
"""
Fake News Detector — Extended
Features:
 A) Model selector (LogReg / Naive Bayes)
 B) Improved UI with badges and probability bars
 C) Token importance visualization (coef / log-prob)
 D) CSV uploader for batch predictions and download
 E) Explanations: SHAP if available, otherwise coefficient-based
 F) Clean + robust preprocessing (no NLTK)
"""

import streamlit as st
import joblib
import os
import re
import numpy as np
import pandas as pd
from io import BytesIO

# Try optional SHAP (for explanations)
try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

st.set_page_config(page_title="Fake News Detector", layout="wide")

# ---------------------------
# Utilities
# ---------------------------
def simple_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # remove urls, punctuation, extra whitespace
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_models(models_dir="models"):
    files = {}
    vec_path = os.path.join(models_dir, "tfidf_vectorizer.joblib")
    lr_path = os.path.join(models_dir, "logistic_regression.joblib")
    nb_path = os.path.join(models_dir, "naive_bayes.joblib")
    # load vectorizer
    vect = None
    lr = None
    nb = None
    missing = []
    try:
        vect = joblib.load(vec_path)
    except Exception:
        missing.append(vec_path)
    try:
        lr = joblib.load(lr_path)
    except Exception:
        missing.append(lr_path)
    try:
        nb = joblib.load(nb_path)
    except Exception:
        # NB optional; it's OK if missing
        nb = None
    return vect, lr, nb, missing

def predict_with_model(model, vectorizer, texts):
    cleaned = [simple_clean(t) for t in texts]
    X = vectorizer.transform(cleaned)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        # assume binary: [prob_real, prob_fake] or [prob_fake, prob_real]?
        # our training used label 1 = FAKE, 0 = REAL; sklearn returns columns in label order.
        # We will return prob_fake = probs[:,1], prob_real = probs[:,0]
        prob_real = probs[:, 0]
        prob_fake = probs[:, 1]
        pred_label = np.where(prob_real >= 0.5, "REAL", "FAKE")
    else:
        # fallback: use predict
        preds = model.predict(X)
        pred_label = np.where(preds == 0, "REAL", "FAKE")
        prob_fake = np.zeros(len(pred_label))
        prob_real = np.zeros(len(pred_label))
    return pred_label, prob_real, prob_fake, X

def top_tokens_for_instance(vectorizer, X_row, model, topn=10, model_name="LogReg"):
    """
    Return DataFrame with tokens and contribution score (positive -> REAL, negative -> FAKE for LogReg).
    For Logistic Regression, we multiply tfidf * coef.
    For MultinomialNB, we derive contribution from log-prob ratio if possible.
    """
    feat_names = np.array(vectorizer.get_feature_names_out())
    row = X_row.toarray().ravel()
    nonzero_idx = np.where(row > 0)[0]
    if nonzero_idx.size == 0:
        return pd.DataFrame(columns=["token", "score"])

    if model is None:
        # fallback: top tfidf tokens
        idx_sorted = row.argsort()[::-1][:topn]
        tokens = feat_names[idx_sorted]
        scores = row[idx_sorted]
        return pd.DataFrame({"token": tokens, "score": scores})

    if hasattr(model, "coef_"):  # Logistic Regression (1 x n_features)
        # coef_ shape (1, n_feats) or (n_classes, n_feats)
        coef = model.coef_
        # If multiclass, take difference between class 0 and 1 (we expect binary)
        if coef.shape[0] > 1:
            coef = coef[0] - coef[1]
        else:
            coef = coef.ravel()
        # contribution = tfidf * coef
        contrib = row * coef
        # For display, positive means moves toward class 0 (REAL) if coef positive; adjust note in UI
        idx_sorted = np.argsort(contrib)[::-1][:topn]
        tokens = feat_names[idx_sorted]
        scores = contrib[idx_sorted]
        return pd.DataFrame({"token": tokens, "score": scores})

    # For MultinomialNB
    if hasattr(model, "feature_log_prob_"):
        # feature_log_prob_ shape (n_classes, n_features)
        # compute log(prob(fake|token) / prob(real|token)) * tfidf
        flp = model.feature_log_prob_
        if flp.shape[0] == 2:
            # log P(feature|class1) - log P(feature|class0)
            log_ratio = flp[1] - flp[0]
            contrib = row * log_ratio
            idx_sorted = np.argsort(contrib)[::-1][:topn]
            tokens = feat_names[idx_sorted]
            scores = contrib[idx_sorted]
            return pd.DataFrame({"token": tokens, "score": scores})
    # fallback
    idx_sorted = row.argsort()[::-1][:topn]
    tokens = feat_names[idx_sorted]
    scores = row[idx_sorted]
    return pd.DataFrame({"token": tokens, "score": scores})

def df_to_csv_bytes(df: pd.DataFrame):
    out = BytesIO()
    df.to_csv(out, index=False)
    return out.getvalue()

# ---------------------------
# Load models
# ---------------------------
VECT, LR_MODEL, NB_MODEL, MISSING = load_models("models")

# Sidebar
st.sidebar.header("Options")
model_choice = st.sidebar.selectbox("Choose model", ("Logistic Regression", "Naive Bayes"))
threshold = st.sidebar.slider("REAL probability threshold", 0.0, 1.0, 0.50)
top_tokens = st.sidebar.slider("Top tokens to show", 0, 20, 8)
show_shap = st.sidebar.checkbox("Use SHAP explanations (if available)", value=False)

# Page header
st.title("📰 Fake News Detector")
st.markdown("TF-IDF + classical models (demo).")

# Show missing files warning
if len(MISSING) > 0:
    st.warning("Some model files failed to load. Missing: " + ", ".join([os.path.basename(p) for p in MISSING]))
    st.info("Make sure models/tfidf_vectorizer.joblib and models/logistic_regression.joblib exist in the repo.")

# Upload or text input
st.subheader("Single prediction")
text = st.text_area("Paste a headline or article", height=150)

cols = st.columns([1, 1, 1])
with cols[0]:
    btn_predict = st.button("Predict")
with cols[1]:
    st.write("")  # spacer
with cols[2]:
    # quick examples
    if st.button("Load example"):
        text = "breaking: celebrity scandal shocks fans"

# Model selection
if model_choice == "Logistic Regression":
    model = LR_MODEL
else:
    model = NB_MODEL

if VECT is None:
    st.error("Vectorizer not loaded. The app cannot make predictions.")
    st.stop()

if model is None:
    st.warning(f"Selected model ({model_choice}) is not available. Please pick a different model.")
    # allow CSV batch even if model missing? stop for now
    st.stop()

# Prediction section
if btn_predict:
    if not text or not text.strip():
        st.warning("Please enter text to predict.")
    else:
        cleaned = simple_clean(text)
        pred_label, prob_real, prob_fake, X_row = predict_with_model(model, VECT, [text])
        prob_real = prob_real[0]
        prob_fake = prob_fake[0]
        label = pred_label[0]

        # Main result area
        st.markdown("### Prediction: " + (":green[REAL]" if label == "REAL" else ":red[FAKE]"))
        # Probability bars
        prob_df = pd.DataFrame({
            "class": ["REAL", "FAKE"],
            "prob": [float(prob_real), float(prob_fake)]
        }).set_index("class")
        st.bar_chart(prob_df)

        st.markdown(f"**REAL probability:** {prob_real:.4f}  \n**FAKE probability:** {prob_fake:.4f}")

        # Top tokens (contribution)
        st.subheader("Key Tokens")
        tok_df = top_tokens_for_instance(VECT, X_row, model, topn=top_tokens, model_name=model_choice)
        if tok_df.empty:
            st.info("No tokens found in input after vectorization.")
        else:
            # normalize sign: positive -> pushes towards REAL if LR coef positive; show color
            tok_df["abs_score"] = np.abs(tok_df["score"])
            tok_df = tok_df.sort_values("abs_score", ascending=False).reset_index(drop=True)
            tok_df_display = tok_df.copy()
            tok_df_display["score"] = tok_df_display["score"].round(6)
            st.dataframe(tok_df_display[["token", "score"]], height=250)
            # small bar chart of scores
            chart = tok_df_display.head(10).set_index("token")["score"]
            st.bar_chart(chart)

        # SHAP or fallback explanation
        st.subheader("Explanation")
        if show_shap and _HAS_SHAP:
            # compute shap values for logistic regression models (requires dense array)
            try:
                explainer = shap.LinearExplainer(model, VECT)
                cleaned_vec = VECT.transform([cleaned])
                shap_vals = explainer.shap_values(cleaned_vec)
                # shap returns array; for binary, shap_vals[0] etc.
                st.write("SHAP explanation (top features):")
                # build dataframe from SHAP
                fv = pd.DataFrame({
                    "feature": VECT.get_feature_names_out(),
                    "value": cleaned_vec.toarray().ravel(),
                    "shap": shap_vals[0].ravel() if isinstance(shap_vals, (list, tuple)) else shap_vals.ravel()
                })
                fv = fv[fv["value"] != 0].sort_values("shap", key=lambda x: np.abs(x), ascending=False).head(top_tokens)
                st.table(fv[["feature", "value", "shap"]])
            except Exception as e:
                st.write("SHAP explanation failed:", str(e))
                st.write("Falling back to coefficient-based explanation below.")
                st.write(tok_df_display[["token", "score"]].head(top_tokens))
        else:
            if show_shap and not _HAS_SHAP:
                st.info("SHAP not installed in the environment. Showing coefficient/log-prob explanations instead.")
            st.write("Top contributing tokens (model-weighted).")
            if not tok_df.empty:
                st.table(tok_df_display[["token", "score"]].head(top_tokens))

# ---------------------------
# Batch predictions (CSV uploader)
# ---------------------------
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
            st.write(f"Dataset has {len(df)} rows.")
            # make predictions in batches
            texts = df["text"].fillna("").astype(str).tolist()
            labels, prob_reals, prob_fakes, X_all = predict_with_model(model, VECT, texts)
            df_out = df.copy()
            df_out["prediction"] = labels
            df_out["prob_real"] = prob_reals
            df_out["prob_fake"] = prob_fakes
            st.dataframe(df_out.head(50))

            csv_bytes = df_to_csv_bytes(df_out)
            st.download_button("Download results CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")

# ---------------------------
# Footer & credits
# ---------------------------
st.markdown("---")
st.caption("Demo: TF-IDF + classical models. Model and vectorizer loaded from models/*.joblib.")
