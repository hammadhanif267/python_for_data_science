import os
import tempfile

import requests
import streamlit as st
from moviepy.editor import AudioFileClip
from openai import OpenAI
from pytube import YouTube


# Define the transcription functions
def transcribe_audio(client, audio_path):
    with open(audio_path, "rb") as audio_file:
        transcription_response = client.audio.transcriptions.create(
            model="whisper-1", file=audio_file
        )
    return transcription_response.text


def transcribe_uploaded_file(client, audio_file):
    with tempfile.NamedTemporaryFile(
        delete=False, suffix="." + audio_file.name.split(".")[-1]
    ) as tmp_file:
        tmp_file.write(audio_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        transcription_text = transcribe_audio(client, tmp_file_path)
        st.write("Transcription:", transcription_text)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    finally:
        os.remove(tmp_file_path)  # Clean up


# Sidebar for API Key input
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")

# Main app title
st.title("🎙️Speech2Text with Whisper-1🤖")

# Initialize the OpenAI client with the API key from the sidebar
client = OpenAI(api_key=api_key)

# Option for users to select input type: YouTube URL, Upload Audio File, or Use Sample Audio
input_type = st.radio(
    "Select input type", ["Upload Audio File", "Enter YouTube URL", "Use Sample Audio"]
)

if input_type == "Enter YouTube URL":
    youtube_url = st.text_input("Enter a YouTube URL")
    if youtube_url and api_key:
        with st.spinner("Downloading YouTube video and extracting audio..."):
            yt = YouTube(youtube_url)
            video = yt.streams.filter(only_audio=True).first()
            temp_video_path = video.download(output_path=tempfile.gettempdir())
            temp_audio_path = os.path.join(tempfile.gettempdir(), "audio.mp3")

            with AudioFileClip(temp_video_path) as audio_clip:
                audio_clip.write_audiofile(temp_audio_path)
            os.remove(temp_video_path)  # Clean up video file

            transcription_text = transcribe_audio(client, temp_audio_path)
            st.write("Transcription:", transcription_text)
            os.remove(temp_audio_path)  # Clean up audio file

elif input_type == "Upload Audio File":
    audio_file = st.file_uploader(
        "Upload an audio file",
        type=["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"],
    )
    if audio_file is not None and api_key:
        transcribe_uploaded_file(client, audio_file)

elif input_type == "Use Sample Audio":
    sample_audio_url = (
        "https://github.com/Rizwankaka/speech2text-Whisper-1/raw/main/audio.mp3"
    )
    st.markdown(
        f"Download the sample audio file [here]({sample_audio_url}).",
        unsafe_allow_html=True,
    )
    if st.button("Transcribe Sample Audio"):
        with st.spinner("Downloading and transcribing sample audio..."):
            r = requests.get(sample_audio_url)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                tmp_file.write(r.content)
                temp_audio_path = tmp_file.name

            transcription_text = transcribe_audio(client, temp_audio_path)
            st.write("Transcription:", transcription_text)
            os.remove(temp_audio_path)  # Clean up

# Now, place the profile information after the main interactive elements
# Adding the HTML footer
# Profile footer HTML for sidebar
sidebar_footer_html = """
<div style="text-align: left;">
    <p style="font-size: 16px;"><b>Author: 🌟 Hammad Hanif 🌟</b></p>
    <a href="https://github.com/hammadhanif267"><img src="https://img.shields.io/badge/GitHub-Profile-blue?style=for-the-badge&logo=github" alt="GitHub"/></a><br>
    <a href="https://www.linkedin.com/in/hammad-hanif-153a182bb/"><img src="https://img.shields.io/badge/LinkedIn-Profile-blue?style=for-the-badge&logo=linkedin" alt="LinkedIn"/></a><br>
    <a href="mailto:hamadhanif267@gmail.com"><img src="https://img.shields.io/badge/Gmail-Contact%20Me-red?style=for-the-badge&logo=gmail" alt="Gmail"/></a>
    <a href="https://www.facebook.com/profile.php?id=100080146477906"><img src="https://img.shields.io/badge/Facebook-Profile-blue?style=for-the-badge&logo=facebook" alt="Facebook"/></a><br>
</div>
"""

# Render profile footer in sidebar at the "bottom"
st.sidebar.markdown(sidebar_footer_html, unsafe_allow_html=True)


# Set a background image
def set_background_image():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://images.pexels.com/photos/4097159/pexels-photo-4097159.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1);
            background-size: cover;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


set_background_image()

# Set a background image for the sidebar
sidebar_background_image = """
<style>
[data-testid="stSidebar"] {
    background-image: url("https://www.pexels.com/photo/abstract-background-with-green-smear-of-paint-6423446/");
    background-size: cover;
}
</style>
"""

st.sidebar.markdown(sidebar_background_image, unsafe_allow_html=True)

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Custom CSS to inject into the Streamlit app
footer_css = """
<style>
.footer {
    position: fixed;
    right: 0;
    bottom: 0;
    width: auto;
    background-color: transparent;
    color: black;
    text-align: right;
    padding-right: 10px;
}
</style>
"""

# HTML for the footer - replace your credit information here
footer_html = """
<div class="footer">
    <p>Credit: Hammad Hanif | Data Science Enthusiast</p>
    <a href="https://github.com/hammadhanif267">
        <img src="https://img.shields.io/badge/GitHub-Profile-blue?style=for-the-badge&logo=github" alt="GitHub"/>
    </a>
    <a href="https://www.linkedin.com/in/hammad-hanif-153a182bb/">
        <img src="https://img.shields.io/badge/LinkedIn-Profile-blue?style=for-the-badge&logo=linkedin" alt="LinkedIn"/>
    </a>
    <a href="mailto:hamadhanif267@gmail.com">
        <img src="https://img.shields.io/badge/Gmail-Contact%20Me-red?style=for-the-badge&logo=gmail" alt="Gmail"/>
    </a>
    <a href="https://www.facebook.com/profile.php?id=100080146477906">
        <img src="https://img.shields.io/badge/Facebook-Profile-blue?style=for-the-badge&logo=facebook" alt="Facebook"/>
    </a>
</div>
"""

# Centered footer CSS and HTML
centered_footer_css = """
<style>
.center-footer {
    position: fixed;
    left: 50%;
    bottom: 0;
    transform: translateX(-50%);
    background-color: transparent;
    color: black;
    text-align: center;
    padding-bottom: 10px;
    z-index: 9999;
}
</style>
"""

centered_footer_html = """
<div class="center-footer" style="color: white; font-weight: bold;">
    <p style="color: white; font-weight: bold;">Credit: Hammad Hanif | Data Science Enthusiast</p>
    <a href="https://github.com/hammadhanif267">
        <img src="https://img.shields.io/badge/GitHub-Profile-blue?style=for-the-badge&logo=github" alt="GitHub"/>
    </a>
    <a href="https://www.linkedin.com/in/hammad-hanif-153a182bb/">
        <img src="https://img.shields.io/badge/LinkedIn-Profile-blue?style=for-the-badge&logo=linkedin" alt="LinkedIn"/>
    </a>
    <a href="mailto:hamadhanif267@gmail.com">
        <img src="https://img.shields.io/badge/Gmail-Contact%20Me-red?style=for-the-badge&logo=gmail" alt="Gmail"/>
    </a>
    <a href="https://www.facebook.com/profile.php?id=100080146477906">
        <img src="https://img.shields.io/badge/Facebook-Profile-blue?style=for-the-badge&logo=facebook" alt="Facebook"/>
    </a>
</div>
"""

st.markdown(centered_footer_css, unsafe_allow_html=True)
st.markdown(centered_footer_html, unsafe_allow_html=True)
