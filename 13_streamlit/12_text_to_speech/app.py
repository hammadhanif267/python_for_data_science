import os
import tempfile

import pydub
import streamlit as st
from openai import OpenAI


# Function to convert text to speech, modified to explicitly use an API key
def text_to_speech(api_key, text: str, model, voice):
    # Initialize the OpenAI client with the provided API key
    client = OpenAI(api_key=api_key)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
        speech_file_path = tmpfile.name
        response = client.audio.speech.create(model=model, voice=voice, input=text)
        # Stream the audio response to file
        response.stream_to_file(speech_file_path)

        # Return the path to the audio file
        return speech_file_path


def convert_audio_format(input_path, output_path, format):
    audio = pydub.AudioSegment.from_mp3(input_path)
    audio.export(output_path, format=format)

    def increase_pitch(audio_path, semitones):
        audio = pydub.AudioSegment.from_mp3(audio_path)
        new_sample_rate = int(audio.frame_rate * (2.0 ** (semitones / 12.0)))
        pitched_audio = audio._spawn(
            audio.raw_data, overrides={"frame_rate": new_sample_rate}
        )
        pitched_audio = pitched_audio.set_frame_rate(audio.frame_rate)
        # Save to a temporary file and return the path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
            pitched_audio.export(tmpfile.name, format="mp3")
            return tmpfile.name


# Streamlit UI setup
st.title("🔊 Text to Speech Converter 📝")
st.image(
    "https://miro.medium.com/v2/resize:fit:1400/format:webp/1*uuVodlqRR5ryF4dRf9TQRg.png",
    # caption="Text to Speech Converter",
    width=400,
)
st.markdown(
    """
This app converts text to speech using OpenAI's tts-1 or tts-1-hd model.
Please enter your OpenAI API key on sidebar. **Do not share your API key with others.**
"""
)

# Input for OpenAI API key
api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")

# Select box for model selection
model = st.sidebar.selectbox("Select Model", ["tts-1", "tts-1-hd"])

# Select box for voice selection
voice = st.sidebar.selectbox(
    "Select Voice", ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
)

# Select box for format selection
format = st.sidebar.selectbox("Select Format", ["mp3", "opus", "aac", "flac", "wav"])

# Text input from user
user_input = st.text_area(
    "Enter text to convert to speech", "Hello, welcome to our text to speech converter!"
)

if st.button("Convert"):
    if not api_key:
        st.error("API key is required to convert text to speech.")
    else:
        with st.spinner("Converting text to speech..."):
            try:
                mp3_speech_path = text_to_speech(api_key, user_input, model, voice)

                if format != "mp3":
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=f".{format}"
                    ) as tmpfile:
                        convert_audio_format(mp3_speech_path, tmpfile.name, format)
                        speech_path = tmpfile.name
                    os.remove(mp3_speech_path)
                else:
                    speech_path = mp3_speech_path

                # Display a link to download the audio file
                st.audio(open(speech_path, "rb"), format=format)
                st.markdown(
                    f"[Download {format.upper()} file]({speech_path})",
                    unsafe_allow_html=True,
                )

                # Clean up: delete the temporary file after use
                os.remove(speech_path)
            except Exception as e:
                st.error(f"An error occurred: {e}")


# Adding the HTML footerP
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
