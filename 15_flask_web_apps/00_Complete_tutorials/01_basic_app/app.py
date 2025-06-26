from flask import Flask

app = Flask(__name__)  # create an instance of the Flask class


@app.route("/")  # route() decorator to tell Flask what URL should call the fun
def hello_world():
    return "Hello, World! How are you"


if __name__ == "__main__":
    app.run(debug=True)  # run the app on the local development server
