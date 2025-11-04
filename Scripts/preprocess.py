import os
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from dataLoader import load_raw_data

nltk.download('stopwords')
nltk.download('wordnet')


train_df, test_df = load_raw_data()

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)


train_df['cleaned_review'] = train_df['review'].astype(str).apply(clean_text)
test_df['cleaned_review'] = test_df['review'].astype(str).apply(clean_text)

train_df['sentiment'] = train_df['review'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
test_df['sentiment'] = test_df['review'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

train_df['review_length'] = train_df['cleaned_review'].apply(lambda x: len(x.split()))
test_df['review_length'] = test_df['cleaned_review'].apply(lambda x: len(x.split()))


base_path = os.path.dirname(os.path.abspath(__file__))
train_clean_path = os.path.join(base_path, "../data/train_cleaned.csv")
test_clean_path = os.path.join(base_path, "../data/test_cleaned.csv")

train_df.to_csv(train_clean_path, index=False)
test_df.to_csv(test_clean_path, index=False)

print("✅ Preprocessing complete. Cleaned data saved at:")
print(train_clean_path)
print(test_clean_path)
