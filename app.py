# app.py
import streamlit as st
import joblib, os, re
import numpy as np
from PIL import Image

# Try to import nltk functions and ensure required data is available
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords

    # Ensure required corpora exist (download only if missing)
    for corpus in ("stopwords", "punkt"):
        try:
            nltk.data.find(f"corpora/{corpus}")
        except LookupError:
            nltk.download(corpus, quiet=True)

except Exception:
    word_tokenize = None
    stopwords = None


st.set_page_config(page_title="Fake News Detector", layout="wide")
st.title("📰 Fake News Detector")
st.markdown("TF-IDF + Logistic Regression (demo).")

MODELS_DIR = "models"
VEC_FILE = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
LR_FILE = os.path.join(MODELS_DIR, "logistic_regression.joblib")
NB_FILE = os.path.join(MODELS_DIR, "naive_bayes.joblib")  # optional

@st.cache_resource
def load_models():
    vec = joblib.load(VEC_FILE) if os.path.exists(VEC_FILE) else None
    lr = joblib.load(LR_FILE) if os.path.exists(LR_FILE) else None
    nb = joblib.load(NB_FILE) if os.path.exists(NB_FILE) else None
    return vec, lr, nb

vectorizer, lr_model, nb_model = load_models()

def simple_clean(text):
    t = str(text).lower()
    t = re.sub(r"http\S+|www\S+|https\S+", " ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if word_tokenize and stopwords:
        sw = set(stopwords.words("english"))
        toks = [w for w in word_tokenize(t) if w not in sw and len(w)>1]
        return " ".join(toks)
    return t

def top_features(vectorizer, model, text, topn=8):
    try:
        fn = np.array(vectorizer.get_feature_names_out())
        coef = model.coef_[0]
        x = vectorizer.transform([text]).toarray()[0]
        scores = x * coef
        idx = np.argsort(scores)[-topn:][::-1]
        return list(fn[idx])
    except Exception:
        return []

# Sidebar
st.sidebar.header("Options")
model_choice = st.sidebar.selectbox("Model", ["LogReg (original)", "Naive Bayes (if available)"])
threshold = st.sidebar.slider("REAL prob threshold", 0.1, 0.9, 0.5, 0.01)
top_k = st.sidebar.slider("Top tokens to show", 0, 12, 6)

# Metrics/images autodiscovery
metrics_paths = [
    os.path.join(MODELS_DIR, "baseline_metrics.csv"),
    os.path.join("code files", "matrix", "baseline_metrics.csv"),
    "baseline_metrics.csv"
]
confusion_paths = [
    os.path.join(MODELS_DIR, "confusion_lr.png"),
    os.path.join("code files", "matrix", "confusion_lr.png"),
    "confusion_lr.png"
]
roc_paths = [
    os.path.join(MODELS_DIR, "roc_curves.png"),
    os.path.join("code files", "matrix", "roc_curves.png"),
    "roc_curves.png"
]

def find_first(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

metrics_file = find_first(metrics_paths)
confusion_file = find_first(confusion_paths)
roc_file = find_first(roc_paths)

# Main layout
left, right = st.columns([2,1])

with left:
    st.subheader("Paste a headline or article")
    text = st.text_area("Input text", height=220)
    if st.button("Check News"):
        if not text or not text.strip():
            st.warning("Please enter some text.")
        else:
            cleaned = simple_clean(text)
            if model_choice.startswith("Naive") and nb_model is not None:
                vec = vectorizer
                clf = nb_model
                mname = "Naive Bayes"
            else:
                vec = vectorizer
                clf = lr_model
                mname = "Logistic Regression"

            if vec is None or clf is None:
                st.error("Model files missing. Put tfidf_vectorizer.joblib and logistic_regression.joblib in ./models/")
            else:
                Xv = vec.transform([cleaned])
                try:
                    prob = float(clf.predict_proba(Xv)[0][1])
                except Exception:
                    # fallback if classifier doesn't support predict_proba
                    pred = clf.predict(Xv)[0]
                    prob = 1.0 if pred == 1 else 0.0
                label = "REAL" if prob >= threshold else "FAKE"
                if label == "REAL":
                    st.success(f"{label}  — Prob REAL: {prob:.3f}")
                else:
                    st.error(f"{label}  — Prob REAL: {prob:.3f}")
                st.write(f"Model: {mname}")
                if top_k>0:
                    tops = top_features(vec, clf, cleaned, topn=top_k)
                    if tops:
                        st.write("Top tokens:", ", ".join(tops))

with right:
    st.subheader("Model Metrics")
    if metrics_file:
        try:
            import pandas as pd
            dfm = pd.read_csv(metrics_file)
            st.table(dfm)
        except Exception:
            st.info("Found metrics file but couldn't read it.")
    else:
        st.info("No baseline_metrics.csv found (optional). Place it in ./models or code files/matrix.")
    if confusion_file:
        st.image(Image.open(confusion_file), caption="Confusion Matrix")
    if roc_file:
        st.image(Image.open(roc_file), caption="ROC Curves")

st.markdown("---")
st.subheader("Batch predict (CSV)")
st.write("Upload a CSV with `text` or `clean_text` column; will return pred_proba_real and pred_label.")
uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded:
    import pandas as pd, numpy as np
    df = pd.read_csv(uploaded)
    if 'clean_text' in df.columns:
        texts = df['clean_text'].astype(str).fillna('')
    elif 'text' in df.columns:
        texts = df['text'].astype(str).fillna('')
    else:
        st.error("CSV must contain 'text' or 'clean_text' column.")
        texts = None
    if texts is not None:
        vec = vectorizer
        clf = lr_model if lr_model is not None else nb_model
        if vec is None or clf is None:
            st.error("Model artifacts missing.")
        else:
            cleaned_texts = [simple_clean(t) for t in texts]
            Xv = vec.transform(cleaned_texts)
            probs = clf.predict_proba(Xv)[:,1]
            preds = np.where(probs >= threshold, "REAL", "FAKE")
            out = df.copy()
            out['pred_proba_real'] = probs
            out['pred_label'] = preds
            st.dataframe(out.head(10))
            csv_bytes = out.to_csv(index=False).encode('utf-8')
            st.download_button("Download predictions", data=csv_bytes, file_name="predictions.csv")
st.caption("Put joblib files in ./models/ next to this app.py")
