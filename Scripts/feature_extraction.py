import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import joblib

base_path = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(base_path, "../data/train_cleaned.csv")
test_path = os.path.join(base_path, "../data/test_cleaned.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

train_df['cleaned_review'] = train_df['cleaned_review'].fillna("")
test_df['cleaned_review'] = test_df['cleaned_review'].fillna("")

tfidf = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1,2))
tfidf_matrix = tfidf.fit_transform(train_df['cleaned_review'])

nn_model = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute')
nn_model.fit(tfidf_matrix)

models_dir = os.path.join(base_path, "../models")
os.makedirs(models_dir, exist_ok=True)

joblib.dump(tfidf, os.path.join(models_dir, "tfidf_vectorizer.pkl"))
joblib.dump(nn_model, os.path.join(models_dir, "nn_model.pkl"))

print("TF-IDF and NearestNeighbors model saved successfully")
