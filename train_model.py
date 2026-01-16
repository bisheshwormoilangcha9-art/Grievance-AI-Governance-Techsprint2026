import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv("data/complaints.csv")

X = data["complaint_text"]
y = data["category"]

vectorizer = TfidfVectorizer(stop_words="english")
X_vec = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vec, y)

with open("model.pkl", "wb") as f:
    pickle.dump((model, vectorizer), f)

print("Model trained and saved successfully")
