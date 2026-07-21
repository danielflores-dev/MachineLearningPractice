import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("product_info.csv")
df["price_usd"] = pd.to_numeric(df["price_usd"], errors="coerce")
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df = df.dropna(subset=["rating", "price_usd"])

X = df[["rating", "loves_count"]]  # change this line
y = df[["price_usd"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

predicted = model.predict(X_test)
print(mean_absolute_error(y_test, predicted))