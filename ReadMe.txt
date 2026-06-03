# 📰 Fake News Detection System

A machine learning-powered web application that classifies news articles as **Real** or **Fake** using Natural Language Processing (NLP) techniques and supervised learning models.

## 🚀 Live Demo

https://fakenewsdetector-ufof8jqirphctyhfoa6egd.streamlit.app/

---

## 📌 Features

- Detects whether a news article is Real or Fake
- Text preprocessing and cleaning pipeline
- TF-IDF based feature extraction
- Logistic Regression classifier
- Interactive Streamlit web interface
- Fast real-time predictions
- Model persistence using Joblib

---

## 🛠️ Tech Stack

### Machine Learning
- Python
- Scikit-Learn
- Logistic Regression
- TF-IDF Vectorization

### Data Processing
- Pandas
- NumPy
- Regular Expressions (Regex)
- NLTK

### Deployment & UI
- Streamlit

### Model Storage
- Joblib

---

## 📂 Project Structure

```text
FakeNewsDetection/
│
├── app.py                      # Streamlit application
├── train_model.py              # Model training script
├── requirements.txt
│
├── data/
│   ├── Fake.csv
│   └── True.csv
│
├── models/
│   ├── tfidf_vectorizer.joblib
│   └── logistic_regression.joblib
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_train_models.ipynb
│   └── 03_visualization.ipynb
│
└── README.md
```

---

## ⚙️ Machine Learning Pipeline

### 1. Data Collection
- Real news dataset
- Fake news dataset

### 2. Data Preprocessing
- Lowercase conversion
- URL removal
- Special character removal
- Whitespace normalization

### 3. Feature Engineering

TF-IDF Vectorization:
- Unigrams and Bigrams
- Maximum Features: 15,000
- Minimum Document Frequency: 3

### 4. Model Training

Logistic Regression Classifier:
- Max Iterations: 2000
- Stratified Train-Test Split
- Binary Classification

### 5. Prediction

User Input → Preprocessing → TF-IDF Transformation → Model Prediction

---

## 📊 Model Performance

| Metric | Value |
|----------|----------|
| Accuracy | ~98% |
| Precision | High |
| Recall | High |
| F1 Score | High |

*Results may vary depending on dataset version and train-test split.*

---

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/Nitya-sigadapu/Fake_news_detector.git
cd Fake_news_detector
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 💡 Future Improvements

- Transformer-based models (BERT, RoBERTa)
- Explainable AI visualizations
- News source credibility scoring
- Multi-language support
- Model comparison dashboard
- Real-time news verification APIs

---

## 📸 Application Workflow

1. Enter a news article or headline.
2. Click **Predict**.
3. The text is preprocessed and vectorized.
4. The trained model classifies the article.
5. Prediction is displayed instantly.

---

## 👨‍💻 Author

**Nitya Sigadapu**

GitHub:
https://github.com/Nitya-sigadapu

---
