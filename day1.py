import pandas as pd


data = {
    "product": ["cleanser", "moisturizer", "sunscreen"],
    "price": [12, 25, 18],
    "rating": [4.5, 4.8, 4.2]
}

df = pd.DataFrame(data)
print(df)

print(df["rating"].mean())   
print(df[df["price"] < 20])  