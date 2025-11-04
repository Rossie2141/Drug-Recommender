import os
import pandas as pd
import joblib
import numpy as np

base_path = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(base_path, "../data/train_cleaned.csv")
train_df = pd.read_csv(train_path)

tfidf = joblib.load(os.path.join(base_path, "../models/tfidf_vectorizer.pkl"))
nn_model = joblib.load(os.path.join(base_path, "../models/nn_model.pkl"))

def recommend_drugs(user_input, condition=None, top_n=5):
    user_vec = tfidf.transform([user_input])
    distances, indices = nn_model.kneighbors(user_vec, n_neighbors=50)  # Get more to filter
    
    recommended_drugs = []
    for idx in indices[0]:
        row = train_df.iloc[idx]
        if condition and str(row['condition']).lower() != condition.lower():
            continue
        if row['sentiment'] <= 0:
            continue
        recommended_drugs.append(row['drugName'])
        if len(recommended_drugs) >= top_n:
            break

    if not recommended_drugs:
        return ["No positively reviewed drugs found for this input."]
    return recommended_drugs

if __name__ == "__main__":
    print("Enter your symptom or review text:")
    user_input = input()
    print("Enter condition (optional, press Enter to skip):")
    condition_input = input().strip() or None
    
    recommendations = recommend_drugs(user_input, condition=condition_input, top_n=5)
    
    print("\nTop Drug Recommendations:")
    for idx, drug in enumerate(recommendations, 1):
        print(f"{idx}. {drug}")
