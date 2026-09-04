import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

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

X = balanced["review_text"]
y = balanced["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

feature_names = vectorizer.get_feature_names_out()


model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

coefficients = pd.Series(model.coef_[0], index = feature_names)

predicted = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, predicted))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predicted))
print(coefficients.sort_values(ascending=False).head(15))
print(coefficients.sort_values().head(15))