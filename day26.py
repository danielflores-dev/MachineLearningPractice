import pandas as pd

df1 = pd.read_csv("reviews_0-250.csv", low_memory=False)
df2 = pd.read_csv("reviews_250-500.csv", low_memory=False)
df3 = pd.read_csv("reviews_500-750.csv", low_memory=False)
df4 = pd.read_csv("reviews_750-1250.csv", low_memory=False)
df5 = pd.read_csv("reviews_1250-end.csv", low_memory=False)

reviews = pd.concat([df1, df2, df3, df4, df5])

print(reviews.shape)
print(reviews.columns.tolist())
print(reviews.head())