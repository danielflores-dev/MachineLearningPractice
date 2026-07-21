import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("product_info.csv")
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df["price_usd"] = pd.to_numeric(df["price_usd"], errors="coerce")
df["loves_count"] = pd.to_numeric(df["loves_count"], errors="coerce")
df = df.dropna(subset=["rating", "price_usd", "loves_count"])

X = df[["rating", "loves_count"]]
y = df["price_usd"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
scores = cross_val_score(model, X_scaled, y, cv=5, scoring="neg_mean_absolute_error")

print("Scores per fold:", -scores)
print("Average MAE:", -scores.mean())