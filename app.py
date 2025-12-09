# app.py

import streamlit as st
import joblib
import os, re
import numpy as np
from PIL import Image

# --------------------------------------------------------------------
# NLTK Import + Safe Auto-Download of Required Corpora
# --------------------------------------------------------------------
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords

    # Ensure required corpora exist (stopwords + punkt)
    REQUIRED = {
        "stopwords": "corpora/stopwords",
        "punkt": "tokenizers/punkt",
    }

    for corpus, path in REQUIRED.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(corpus, quiet=True)

except Exception:
    word_tokenize = None
    stopwords = None


# --------------------------------------------------------------------
# Fallback Tokenizer (works even if NLTK fails)
# --------------------------------------------------------------------
def fallback_tokenize(text):
    # simple regex tokenizer
    return re.findall(r"\b\w+\b", text.lower())


# --------------------------------------------------------------------
# Text Cleaning Function
# --------------------------------------------------------------------
def simple_clean(text):
    text = text.lower()

    # Stopwords list
    if stopwords:
        sw = set(stopwords.words("english"))
    else:
        # minimal fallback stopwords
        sw = {"the", "and", "is", "a", "an", "to", "of", "in", "on"}

    # Tokenize
    if word_tokenize:
        try:
            tokens = word_tokenize(text)
        except LookupError:
            tokens = fallback_tokenize(text)
    else:
        tokens = fallback_tokenize(text)

    # Filter and clean tokens
    cleaned = []
    for w in tokens:
        w = re.sub(r"[^a-z0-9]", "", w)
        if w and w not in sw and len(w) > 1:
            cleaned.append(w)

    return " ".join(cleaned)


# --------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------
st.set_page_config(page_title="Fake News Detector", layout="wide")
st.title("📰 Fake News Detector")
st.markdown("**TF-IDF + Logistic Regression (demo).**")


# --------------------------------------------------------------------
# Load Models
# --------------------------------------------------------------------
MODELS_DIR = "models"

VEC_FILE = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
LR_FILE  = os.path.join(MODELS_DIR, "logistic_regression.joblib")

vectorizer = joblib.load(VEC_FILE)
lr_model   = joblib.load(LR_FILE)


# --------------------------------------------------------------------
# Sidebar Options
# --------------------------------------------------------------------
st.sidebar.header("Options")
threshold = st.sidebar.slider("REAL probability threshold", 0.0, 1.0, 0.50)
top_tokens = st.sidebar.slider("Top tokens to show", 0, 20, 5)


# --------------------------------------------------------------------
# User Input
# --------------------------------------------------------------------
text = st.text_area("Paste a headline or article", height=150)

if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        cleaned = simple_clean(text)
        vec = vectorizer.transform([cleaned])

        prob_real = lr_model.predict_proba(vec)[0][0]
        prob_fake = lr_model.predict_proba(vec)[0][1]
        label = "REAL" if prob_real >= threshold else "FAKE"

        st.subheader(f"Prediction: **{label}**")
        st.write(f"**REAL probability:** {prob_real:.4f}")
        st.write(f"**FAKE probability:** {prob_fake:.4f}")

        # top TF-IDF words
        if top_tokens > 0:
            st.write("### Key Tokens")
            feature_names = np.array(vectorizer.get_feature_names_out())
            sorted_idx = vec.toarray()[0].argsort()[::-1]
            tokens = feature_names[sorted_idx][:top_tokens]
            st.write(tokens)


# --------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------
st.markdown("---")
st.caption("Fake News Detector demo using Logistic Regression + TF-IDF.")
