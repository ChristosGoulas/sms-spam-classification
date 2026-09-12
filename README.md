# 📩 SMS Spam Classification with Machine Learning

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Natural Language Processing (NLP) binary classification pipeline built with **Python** and **scikit-learn** to detect spam messages in SMS text data.

---

## 📌 Project Overview

Spam SMS messages are a persistent security and user-experience challenge. This project builds and benchmarks machine learning classifiers to automatically identify unsolicited spam messages (`True`) versus legitimate ham messages (`False`).

### Key Highlights
- **Exploratory Data Analysis (EDA)**: Analyzed message length distributions, character counts, and class imbalance.
- **NLP Vectorization**: Compared Bag-of-Words (`CountVectorizer`) against TF-IDF (`TfidfVectorizer`) with English stop-word removal.
- **Classifier Benchmarking**: Evaluated **Multinomial Naive Bayes**, **Logistic Regression**, and **Support Vector Machines (Linear SVC)**.
- **Rigorous Validation**: Implemented an 80/20 stratified train-validation split to eliminate data leakage and evaluate real-world generalization.

---

## 📊 Benchmark Results

Evaluated on a stratified holdout validation set ($N = 890$):

| Vectorizer | Classifier | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **CountVectorizer** | **Multinomial Naive Bayes** | **98.65%** | **96.88%** | **91.18%** | **93.94%** |
| **TfidfVectorizer** | **Support Vector Machine (Linear)** | **98.43%** | **97.83%** | **88.24%** | **92.78%** |
| **CountVectorizer** | **Logistic Regression** | **98.20%** | **98.88%** | **85.29%** | **91.58%** |
| **TfidfVectorizer** | **Multinomial Naive Bayes** | **97.08%** | **100.00%** | **75.00%** | **85.71%** |

> **Selected Model**: **Multinomial Naive Bayes with CountVectorizer** achieved the highest F1-Score (**93.94%**) and high recall on spam messages.

---

## 📁 Repository Structure

```text
spam-or-not/
├── data/
│   ├── training_data.txt       # Labeled SMS messages (Tab-separated: Label, Text)
│   ├── test_data.txt           # Unlabeled SMS messages (Tab-separated: Id, Text)
│   ├── random_submission.csv   # Kaggle-style submission format reference
│   └── submission.csv          # Generated predictions on unseen test data
├── notebooks/
│   └── sms_spam_classification.ipynb  # Interactive NLP pipeline & model benchmarks
├── .gitignore                  # Git exclusion rules
├── LICENSE                     # MIT License
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have **Python 3.10+** installed.

### 2. Installation

Clone the repository and set up a virtual environment:

```bash
# Clone the repository
git clone https://github.com/ChristosGoulas/spam-or-not.git
cd spam-or-not

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 💡 Usage

Launch the Jupyter Notebook to run EDA, train models, and reproduce predictions:

```bash
jupyter notebook notebooks/sms_spam_classification.ipynb
```

The final predictions for unseen test messages will be saved to `data/submission.csv`.

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more details.
