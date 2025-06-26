import pickle

import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the expanded JSON data
df = pd.read_json("Data/House_Price.json")

# Features and target
X = df["Area(in sq. ft)"].values.reshape(-1, 1)
y = df["Price(in Rs.)"].values.reshape(-1, 1)

# Model training
model = LinearRegression()
model.fit(X, y)

# Save the trained model
pickle.dump(model, open("model.pkl", "wb"))
