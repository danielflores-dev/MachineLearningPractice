import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("product_info.csv")
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df["price_usd"] = pd.to_numeric(df["price_usd"], errors="coerce")
df["loves_count"] = pd.to_numeric(df["loves_count"], errors="coerce")
df = df.dropna(subset=["rating", "price_usd", "loves_count"])

encoded = pd.get_dummies(df, columns=["primary_category"])
feature_cols = ["rating", "loves_count", "primary_category_Fragrance"]

X = encoded[feature_cols]
y = encoded["price_usd"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = RandomForestRegressor(random_state=1)
model.fit(X_train, y_train)

importances = pd.Series(model.feature_importances_, index=feature_cols)
print(importances.sort_values(ascending=False))

from sklearn.metrics import mean_absolute_error

predicted = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, predicted))