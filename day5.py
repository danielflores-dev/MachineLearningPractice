import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("product_info.csv")
df["price_usd"] = pd.to_numeric(df["price_usd"], errors="coerce")
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df = df.dropna(subset=["rating", "price_usd"])

X = df[["rating"]]
y= df[["price_usd"]]

model = LinearRegression ()
model.fit(X,y)

predicted = model.predict([[4.5]])
print(predicted)