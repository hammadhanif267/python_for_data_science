# train_model.py
import joblib
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the dataset
tips = sns.load_dataset('tips')

# For simplicity, we'll use 'total_bill' and 'size' as features
# Convert categorical variables to dummy variables if you use them
X = tips[['total_bill', 'size']]
y = tips['tip']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'tip_predictor_model.pkl')

print("Model trained and saved.")