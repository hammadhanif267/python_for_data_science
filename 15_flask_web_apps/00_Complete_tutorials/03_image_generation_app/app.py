import openai
from flask import Flask, redirect, render_template, request, url_for

# Create a Flask app
app = Flask(__name__)

# Load your OpenAI API key (hardcoded for testing purposes)
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_image(prompt):
    client = openai.OpenAI(api_key=openai_api_key)

    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )

    return response.data[0].url


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        prompt = request.form.get("prompt")

        if prompt:
            image_url = generate_image(prompt)
            if image_url:
                return render_template("index.html", image_url=image_url)
            else:
                return render_template(
                    "index.html", error="Failed to generate image. Please try again."
                )
        else:
            return render_template("index.html", error="Please enter a description.")

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
