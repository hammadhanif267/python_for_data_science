import pickle

import pandas as pd
from sklearn.linear_model import LinearRegression

# Load JSON data
df = pd.read_json('Data/Student_Marks.json')

# Features and labels
X = df['Study_Hours'].values.reshape(-1, 1)
y = df['Marks'].values.reshape(-1, 1)

# Model training
model = LinearRegression()
model.fit(X, y)

# Save model
pickle.dump(model, open('model.pkl', 'wb'))
