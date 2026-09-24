import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from nltk.stem import PorterStemmer
import joblib
import re

stemmer = PorterStemmer()

def stem_text(text):
    words=text.split()
    stemmed = []
    for word in words:
        stemmed.append(stemmer.stem(word))
    return " ".join(stemmed)


df1 = pd.read_csv("reviews_0-250.csv", low_memory=False)
df2 = pd.read_csv("reviews_250-500.csv", low_memory=False)
df3 = pd.read_csv("reviews_500-750.csv", low_memory=False)
df4 = pd.read_csv("reviews_750-1250.csv", low_memory=False)
df5 = pd.read_csv("reviews_1250-end.csv", low_memory=False)

reviews = pd.concat([df1, df2, df3, df4, df5])
reviews = reviews[["review_text", "rating"]].dropna()
reviews = reviews[reviews["rating"] != 3]
reviews["sentiment"] = (reviews["rating"] >= 4).astype(int)

positive = reviews[reviews["sentiment"] == 1].sample(114061, random_state=42)
negative = reviews[reviews["sentiment"] == 0]
balanced = pd.concat([positive, negative]).sample(frac=1, random_state=42)
balanced["review_text"] = balanced["review_text"].str.lower()
balanced["review_text"] = balanced["review_text"].str.replace(r'[^a-zA-Z\s]', '', regex=True)
balanced["review_text"] = balanced["review_text"].apply(stem_text)

X = balanced["review_text"]
y = balanced["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

feature_names = vectorizer.get_feature_names_out()


model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

coefficients = pd.Series(model.coef_[0], index = feature_names)

predicted = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, predicted))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predicted))
print(coefficients.sort_values(ascending=False).head(15))
print(coefficients.sort_values().head(15))


def predict_sentiment(text):
    loaded_model = joblib.load("sentiment_model.pkl")
    loaded_vectorizer = joblib.load("vectorizer.pkl")

    cleaned = text.lower()
    cleaned = re.sub(r'[^a-zA-Z\s]', '', cleaned)
    cleaned = stem_text(cleaned)   

    vector = loaded_vectorizer.transform([cleaned])
    prediction = loaded_model.predict(vector)         

    if prediction[0] == 1:
        return "positive"
    else:
        return "negative"

print(predict_sentiment("this broke me out immediately, total waste of money"))
print(predict_sentiment("obsessed with this, my skin has never looked better"))
print(predict_sentiment("wanted to love this but it did nothing"))
print(predict_sentiment("not bad at all actually"))
print(predict_sentiment("expensive but worth every penny"))