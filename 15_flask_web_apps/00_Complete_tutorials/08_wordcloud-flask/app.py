import os
from collections import Counter

import docx
import matplotlib.pyplot as plt
from flask import (
    Flask,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from PyPDF2 import PdfReader
from werkzeug.utils import secure_filename
from wordcloud import WordCloud

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"

# Ensure upload folder exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


def extract_text_from_pdf(filepath):
    pdf_reader = PdfReader(filepath)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text


def extract_text_from_docx(filepath):
    doc = docx.Document(filepath)
    return " ".join([paragraph.text for paragraph in doc.paragraphs])


def generate_and_save_wordcloud(text, format):
    wordcloud = WordCloud(width=1200, height=800, background_color="white").generate(
        text
    )
    plt.figure(figsize=(12, 8), facecolor=None)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)

    img_path = os.path.join(app.config["UPLOAD_FOLDER"], f"wordcloud.{format}")
    plt.savefig(img_path, format=format, bbox_inches="tight", pad_inches=0)
    plt.close()
    return img_path, wordcloud.words_


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files["file"]
        filename = secure_filename(file.filename)

        if filename == "":
            return "No selected file"

        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)
        print(f"File saved to {file_path}")

        if filename.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        elif filename.endswith(".docx"):
            text = extract_text_from_docx(file_path)
        else:
            text = ""

        if not text.strip():
            return "File contains no readable text."

        format = request.form.get("format", "png")
        img_path, frequencies = generate_and_save_wordcloud(text, format)

        # Convert frequencies to {word: freq} dict (multiply by 100 for % appearance)
        freq_dict = {word: freq * 100 for word, freq in frequencies.items()}

        return render_template(
            "result.html", filename=f"wordcloud.{format}", frequencies=freq_dict
        )

    return render_template("index.html")


@app.route("/result/<filename>")
def result(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/download/<filename>")
def download(filename):
    return send_from_directory(
        app.config["UPLOAD_FOLDER"], filename, as_attachment=True
    )


if __name__ == "__main__":
    print(f'Upload folder is set to: {os.path.abspath(app.config["UPLOAD_FOLDER"])}')
    app.run(debug=True)
