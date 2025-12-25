import streamlit as st
from TTS.api import TTS
import os
from pathlib import Path
import pickle
from datetime import datetime

st.set_page_config(page_title="🎙️ Voice Cloner", layout="wide")

MODELS_DIR = Path("user_models")
MODELS_DIR.mkdir(exist_ok=True)

@st.cache_resource
def load_model():
    return TTS("tts_models/en/ljspeech/tacotron2-DDC")

tts = load_model()

tab1, tab2 = st.tabs(["👤 Train Voice", "🎤 Generate Speech"])

with tab1:
    st.header("Train Voice Model")
    username = st.text_input("Username")
    audio_file = st.file_uploader("Upload voice", type=['wav','mp3'])
    
    if st.button("Train") and username and audio_file:
        audio_path = MODELS_DIR / f"{username}.wav"
        with open(audio_path, "wb") as f:
            f.write(audio_file.read())
        
        st.success(f"{username} model saved!")
        st.rerun()

with tab2:
    st.header("Generate Speech")
    models = [f.stem for f in MODELS_DIR.glob("*.wav")]
    
    if models:
        voice = st.selectbox("Select voice", models)
        text = st.text_area("Text")
        
        if st.button("Generate"):
            output_path = "output.wav"
            tts.tts_to_file(text=text, speaker_wav=str(MODELS_DIR/f"{voice}.wav"), file_path=output_path)
            
            with open(output_path, "rb") as f:
                st.audio(f.read())
                st.download_button("Download", f.read())
    else:
        st.info("No models. Train first!")
