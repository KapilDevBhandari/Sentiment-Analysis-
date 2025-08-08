# 🧠 Sentiment Analysis with Logistic Regression

This project applies machine learning techniques to perform **sentiment analysis** on a dataset of 1.6 million preprocessed tweets. It uses **TF-IDF vectorization** and a **Logistic Regression** classifier to categorize tweets into Positive, Negative, or Neutral sentiments.

## 📁 Dataset
The dataset used is the `training.1600000.processed.noemoticon.csv` which contains:
- 1.6 million tweets
- Sentiment labels (0 = negative, 2 = neutral, 4 = positive)

## 🛠️ Features
- Preprocessing with NLTK (stopword removal, stemming, URL & mention cleanup)
- Feature extraction using **TF-IDF Vectorizer**
- Model training using **Logistic Regression**
- Evaluation with **Confusion Matrix** and **Classification Report**
- Visualizations:
  - Confusion Matrix
  - Precision/Recall/F1 Bar Chart
  - Sentiment Distribution Pie Chart

## 📦 Libraries Used
- Python 🐍
- Pandas
- NLTK
- Scikit-learn
- Matplotlib
- Seaborn

## 📊 Results
The model achieved solid performance across classes with an F1-score of:
- **Positive**: 0.78
- **Negative**: 0.77

## 🔍 How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
