import pandas as pd

df = pd.read_csv("product_info.csv")

df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df["price_usd"] = pd.to_numeric(df["price_usd"], errors="coerce")

print(df.shape)
print(df["rating"].mean())
print(df["price_usd"].max())
print(df["brand_name"].nunique())