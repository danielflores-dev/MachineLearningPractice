import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("product_info.csv")
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df["price_usd"] = pd.to_numeric(df["price_usd"], errors="coerce")
df["loves_count"] = pd.to_numeric(df["loves_count"], errors="coerce")
df["reviews"] = pd.to_numeric(df["reviews"], errors="coerce")
df["out_of_stock"] = pd.to_numeric(df["out_of_stock"], errors="coerce")
df["online_only"] = pd.to_numeric(df["online_only"], errors="coerce")
df["limited_edition"] = pd.to_numeric(df["limited_edition"], errors="coerce")
df = df.dropna(subset=["rating", "price_usd", "loves_count"])

encoded = pd.get_dummies(df, columns=["primary_category"])

X = encoded[["rating", "loves_count", "primary_category_Fragrance","reviews", "out_of_stock", "online_only", "limited_edition"]].values
y = df["price_usd"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

model = nn.Sequential(
    nn.Linear(7, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64,1)
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for epoch in range(500):
    model.train()
    optimizer.zero_grad()
    predicted = model(X_train)
    loss = loss_fn(predicted, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.2f}")

model.eval()
with torch.no_grad():
    predictions = model(X_test).numpy()

print("MAE:", mean_absolute_error(y_test.numpy(), predictions))
print(df.select_dtypes(include="number").columns.tolist())