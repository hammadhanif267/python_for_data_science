import pickle

import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    study_hours = int(request.form['hours'])
    final_features = np.array([[study_hours]])
    prediction = model.predict(final_features)
    return render_template('index.html', prediction_text=f'Expected Marks: {prediction[0][0]:.2f}')

if __name__ == "__main__":
    app.run(debug=True)
