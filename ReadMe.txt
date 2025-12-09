# FakeNewsDetection

Link for the App
https://fakenewsdetector-ufof8jqirphctyhfoa6egd.streamlit.app/

Simple Fake News Detection demo using TF-IDF + Logistic Regression and Streamlit.

## Quick run (local)
1. Create a Python environment and install dependencies:
## Files
- `app.py` — Streamlit app
- `models/` — contains joblib artifacts (vectorizer + model)
- `notebooks/` — training & EDA notebooks
- `requirements.txt` — pip packages

## Notes
- If you don't want to commit large model files to GitHub, upload them to GitHub Releases and adjust `app.py` to download them at startup.
- For NLTK, run `nltk.download('punkt')` and `nltk.download('stopwords')` once if needed.

2. Ensure these model files exist in `models/`:
- `models/tfidf_vectorizer.joblib`
- `models/logistic_regression.joblib`
3. Run the app:
__pycache__/
.ipynb_checkpoints/
.DS_Store
.env
.venv/
venv/
models/*.joblib
*.pyc
5) models/ folder contents (what should be inside)

Create folder: C:\Users\hp\Desktop\FakeNewsDet\models\

Required files (create by training or copying from your training notebook):

tfidf_vectorizer.joblib — saved TF-IDF vectorizer (joblib.dump(vectorizer, ...))

logistic_regression.joblib — saved logistic regression model (joblib.dump(lr, ...))

Optional but useful:

naive_bayes.joblib — trained NB model

baseline_metrics.csv — small CSV with metrics (model, accuracy, roc_auc, etc.)

confusion_lr.png — confusion matrix plot (PNG)

roc_curves.png — ROC plot (PNG)

If you already trained in a notebook, use the joblib.dump(...) commands to create these files. If not, follow the training snippet below (run in a Jupyter notebook in the project root).

6) notebooks/ folder suggestions

Create C:\Users\hp\Desktop\FakeNewsDet\notebooks\ and put your notebooks there:

01_data_cleaning.ipynb — data loading, EDA, cleaning steps

02_train_models.ipynb — TF-IDF vectorizer training, model training, evaluation, saving joblibs

03_visualization.ipynb — plots and charts for README

Each notebook should contain code cells for the steps, markdown explanations and saved artifacts (plots, CSVs).

7) If you don't yet have the model files — training snippet (one-shot)

Open a Jupyter notebook in project root (or create notebooks/02_train_models.ipynb) and paste this cell to train and save the required joblibs. Adjust the data paths if necessary.
# TRAIN & SAVE MODELS — run in notebooks/ or project root
import pandas as pd, os, re, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

root = r"C:\Users\hp\Desktop\FakeNewsDet"
# If you have data/Fake.csv and data/True.csv
fake_path = os.path.join(root, "data", "Fake.csv")
true_path = os.path.join(root, "data", "True.csv")

df_fake = pd.read_csv(fake_path)
df_true = pd.read_csv(true_path)
df_fake['label'] = 'FAKE'
df_true['label'] = 'REAL'
df = pd.concat([df_fake, df_true], ignore_index=True)

def simple_clean(s):
    s = str(s).lower()
    s = re.sub(r"http\S+|www\S+|https\S+", " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

df['clean_text'] = df['text'].fillna('').apply(simple_clean)

X = df['clean_text']
y = df['label'].map({'FAKE':0,'REAL':1})

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1,2), min_df=3)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

lr = LogisticRegression(max_iter=2000)
lr.fit(X_train_tfidf, y_train)

os.makedirs(os.path.join(root, "models"), exist_ok=True)
joblib.dump(vectorizer, os.path.join(root, "models", "tfidf_vectorizer.joblib"))
joblib.dump(lr, os.path.join(root, "models", "logistic_regression.joblib"))

print("Saved models to", os.path.join(root, "models"))
8) How to run locally (final checklist)

Open Anaconda Prompt, activate your environment (if you created one), and change dir:
cd C:\Users\hp\Desktop\FakeNewsDet
Install dependencies:
pip install -r requirements.txt
Run Streamlit:
streamlit run app.py
git init
git add .
git commit -m "FakeNewsDetection initial"
# create repo on GitHub via web, then:
git remote add origin https://github.com/<your-username>/FakeNewsDet.git
git branch -M main
git push -u origin main
If your model files are large, do not commit them; upload them to GitHub Releases and either (a) leave them out of the repo and provide download instructions in README, or (b) use Git LFS.
