# -*- coding: utf-8 -*-
"""Sentiment Analysis using Logistic Regression"""

# =======================
# 📦 Imports
# =======================
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# =======================
# 📥 Data Loading
# =======================
df = pd.read_csv("/content/training.1600000.processed.noemoticon.csv", encoding='latin-1', header=None)
df.columns = ['target', 'id', 'date', 'query', 'user', 'text']
df['sentiment'] = df['target'].map({0: 'negative', 2: 'neutral', 4: 'positive'})
df = df[['text', 'sentiment']]

# =======================
# 🧹 Preprocessing
# =======================
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_tweet(text):
    text = text.lower()
    text = re.sub(r'@\w+', '', text)             # Remove @mentions
    text = re.sub(r'#', '', text)                # Remove hashtag symbols
    text = re.sub(r'http\S+|www\S+', '', text)   # Remove URLs
    text = re.sub(r'[^\w\s]', '', text)          # Remove punctuation
    text = re.sub(r'\d+', '', text)              # Remove numbers
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['clean_text'] = df['text'].apply(preprocess_tweet)

# =======================
# 🔤 Feature Extraction
# =======================
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['clean_text'])
y = df['sentiment']

print("TF-IDF Matrix shape:", X.shape)

# =======================
# 📊 Train/Test Split
# =======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training size:", X_train.shape)
print("Testing size:", X_test.shape)

# =======================
# 🤖 Model Training
# =======================
model = LogisticRegression(max_iter=1000, solver='saga', n_jobs=-1)
model.fit(X_train, y_train)

# =======================
# 📈 Evaluation
# =======================
y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualize Confusion Matrix
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# =======================
# 📊 Bar Chart: Metrics
# =======================
metrics = {
    'Class': ['Negative', 'Positive'],
    'Precision': [0.79, 0.76],
    'Recall': [0.75, 0.80],
    'F1-score': [0.77, 0.78]
}

df_metrics = pd.DataFrame(metrics).set_index('Class')
df_metrics.plot(kind='bar', figsize=(8, 6))
plt.title('Precision, Recall, and F1-score per Class')
plt.ylabel('Score')
plt.ylim(0.7, 0.85)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# =======================
# 🥧 Pie Chart: Predictions
# =======================
pred_counts = Counter(y_pred)

plt.figure(figsize=(6, 6))
plt.pie(pred_counts.values(), labels=pred_counts.keys(), autopct='%1.1f%%', startangle=140)
plt.title("Distribution of Predicted Sentiments")
plt.axis('equal')
plt.show()
