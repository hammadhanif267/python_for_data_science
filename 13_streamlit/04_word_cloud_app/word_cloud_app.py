import base64
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import PyPDF2
import streamlit as st
from docx import Document
from wordcloud import STOPWORDS, WordCloud


# Functions for file reading
def read_txt(file):
    return file.getvalue().decode("utf-8")


def read_docx(file):
    doc = Document(file)
    return " ".join([para.text for para in doc.paragraphs])


def read_pdf(file):
    pdf = PyPDF2.PdfReader(file)
    return " ".join([page.extract_text() for page in pdf.pages])


# Function to filter out stopwords
def filter_stopwords(text, additional_stopwords=[]):
    words = text.split()
    all_stopwords = STOPWORDS.union(set(additional_stopwords))
    filtered_words = [word for word in words if word.lower() not in all_stopwords]
    return " ".join(filtered_words)


# Function to create download link for plot
def get_image_download_link(buffered, format_):
    image_base64 = base64.b64encode(buffered.getvalue()).decode()
    return f'<a href="data:image/{format_};base64,{image_base64}" download="wordcloud.{format_}">Download Plot as {format_}</a>'


# Function to generate a download link for a DataFrame
def get_table_download_link(df, filename, file_label):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return (
        f'<a href="data:file/csv;base64,{b64}" download="{filename}">{file_label}</a>'
    )


# --- Streamlit UI ---
st.title("Word Cloud Generator")
st.subheader("📁 Upload a PDF, DOCX, or TXT file to generate a word cloud")

# Add custom CSS for advanced styling
st.markdown(
    """
<style>
    body {
        background-color: #f0f4f8;
        color: #333;
    }
    .stButton > button {
        background-color: #6c63ff;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        transition: transform 0.2s, background-color 0.2s;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        background-color: #5a54e1;
    }
    .stTextInput, .stSelectbox, .stMultiSelect {
        background-color: #ffffff;
        border: 1px solid #ccc;
        border-radius: 5px;
    }
    .stMarkdown {
        margin: 10px 0;
    }
    .stFileUploader {
        border: 2px dashed #6c63ff;
        border-radius: 10px;
        padding: 20px;
        background-color: #ffffff;
    }
</style>
""",
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf", "docx"])
st.set_option("deprecation.showPyplotGlobalUse", False)

if uploaded_file:
    file_details = {
        "FileName": uploaded_file.name,
        "FileType": uploaded_file.type,
        "FileSize": uploaded_file.size,
    }
    st.write(file_details)

    # Detect file type and extract text
    if uploaded_file.type == "text/plain":
        text = read_txt(uploaded_file)
    elif uploaded_file.type == "application/pdf":
        text = read_pdf(uploaded_file)
    elif (
        uploaded_file.type
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ):
        text = read_docx(uploaded_file)
    else:
        st.error("File type not supported. Please upload a TXT, PDF, or DOCX file.")
        st.stop()

    # Initial word count
    words = text.split()
    word_count = (
        pd.DataFrame({"Word": words})
        .groupby("Word")
        .size()
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
    )

    # Sidebar: Stopwords settings
    use_standard_stopwords = st.sidebar.checkbox("Use standard stopwords?", True)
    top_words = word_count["Word"].head(50).tolist()
    additional_stopwords = st.sidebar.multiselect(
        "Additional stopwords:", sorted(top_words)
    )

    all_stopwords = (
        STOPWORDS.union(set(additional_stopwords))
        if use_standard_stopwords
        else set(additional_stopwords)
    )
    text = filter_stopwords(text, all_stopwords)

    if text:
        # Word Cloud size
        width = st.sidebar.slider("Select Word Cloud Width", 400, 2000, 1200, 50)
        height = st.sidebar.slider("Select Word Cloud Height", 200, 2000, 800, 50)

        # Generate and show Word Cloud
        st.subheader("Generated Word Cloud")
        fig, ax = plt.subplots(figsize=(width / 100, height / 100))
        wordcloud_img = WordCloud(
            width=width,
            height=height,
            background_color="white",
            max_words=200,
            contour_width=3,
            contour_color="steelblue",
        ).generate(text)
        ax.imshow(wordcloud_img, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

        # Save plot
        format_ = st.selectbox(
            "Select file format to save the plot", ["png", "jpeg", "svg", "pdf"]
        )
        resolution = st.slider("Select Resolution", 100, 500, 300, 50)
        if st.button(f"Save as {format_}"):
            buffered = BytesIO()
            plt.savefig(buffered, format=format_, dpi=resolution)
            st.markdown(
                get_image_download_link(buffered, format_), unsafe_allow_html=True
            )

        # Word Count Table
        st.subheader("Word Count Table")
        words_filtered = text.split()
        word_count_filtered = (
            pd.DataFrame({"Word": words_filtered})
            .groupby("Word")
            .size()
            .reset_index(name="Count")
            .sort_values("Count", ascending=False)
        )
        st.write(word_count_filtered)

        if st.button("Download Word Count Table as CSV"):
            st.markdown(
                get_table_download_link(
                    word_count_filtered, "word_count.csv", "Click Here to Download"
                ),
                unsafe_allow_html=True,
            )

    # Sidebar Info
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Created By**:<br><i class='fab fa-github'></i> <a href='https://github.com/hammadhanif267' target='_blank'>Hammad Hanif</a>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        "<i class='fas fa-envelope'></i> <b>Contact Me</b>: <a href='mailto:hamadhanif267@gmail.com' target='_blank'>hamadhanif267@gmail.com</a>",
        unsafe_allow_html=True,
    )
