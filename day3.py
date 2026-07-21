print("running")
import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    "rating" : [4.2,4.5,4.8],
    "price" : [18,12,25]
}

df = pd.DataFrame(data)

X = df[["rating"]]
y = df[["price"]]

model = LinearRegression()
model.fit(X, y)

predicted = model.predict([[4.2]])
print(predicted)