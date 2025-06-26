import pickle

import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    feature = [int(x) for x in request.form.values()]
    area = feature[0]

    # Validation as per new data range (500 to 5000 sq.ft)
    if area < 500 or area > 5000:
        return render_template(
            "index.html",
            prediction_text="Error: Area should be between 500 and 5000 sq. ft.",
        )

    feature_final = np.array(feature).reshape(-1, 1)
    prediction = model.predict(feature_final)
    return render_template(
        "index.html",
        prediction_text="Price of House will be Rs. {}".format(int(prediction)),
    )


if __name__ == "__main__":
    app.run(debug=True)
