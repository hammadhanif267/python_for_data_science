# http://127.0.0.1:5000/
from flask import Flask, render_template, request
from real_time_predict import predict_image
import os

UPLOAD_FOLDER = 'static/uploads'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    file_path = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            prediction = predict_image(file_path)

    return render_template('index.html', prediction=prediction, file_path=file_path)

if __name__ == '__main__':
    app.run(debug=True)