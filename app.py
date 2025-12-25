"""
🎙️ Multi-User Perfect Voice Cloner
Upload voice → Train model → Select from dropdown → Perfect clone!
"""

import streamlit as st
import torch
from TTS.api import TTS
import os
from pathlib import Path
import pickle
import uuid
import time
from datetime import datetime
import io

# Page config
st.set_page_config(
    page_title="🎙️ Multi-User Voice Cloner", 
    page_icon="🎙️",
    layout="wide"
)

# Models directory
@st.cache_data
def get_models_dir():
    models_dir = Path("user_models")
    models_dir.mkdir(exist_ok=True)
    return models_dir

MODELS_DIR = get_models_dir()

# Load base TTS model
@st.cache_resource
def load_tts_model():
    return TTS("tts_models/multilingual/multi-dataset/xtts_v2")

tts = load_tts_model()
st.success("✅ XTTS-v2 model loaded!")

# Save user model
def save_user_model(username, speaker_wav_path):
    model_path = MODELS_DIR / f"{username}_model.pkl"
    user_model = {
        "speaker_wav": str(speaker_wav_path),
        "username": username,
        "created": datetime.now().isoformat(),
        "id": str(uuid.uuid4())
    }
    with open(model_path, "wb") as f:
        pickle.dump(user_model, f)
    return model_path

# Get all trained models
def get_all_models():
    models = []
    for model_file in MODELS_DIR.glob("*.pkl"):
        try:
            with open(model_file, "rb") as f:
                model_data = pickle.load(f)
            models.append(model_data)
        except:
            continue
    return sorted(models, key=lambda x: x["created"], reverse=True)

# Header
st.markdown("""
# 🎙️ **Multi-User Perfect Voice Cloner**
Upload your voice → Train personal model → Generate speech from ANY voice!
""")

# Main tabs
tab1, tab2 = st.tabs(["👤 Train Voice Model", "🎤 Generate Speech"])

# TAB 1: Train new voice model
with tab1:
    st.header("👤 Train Your Personal Voice Model")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        username = st.text_input("👤 Username", placeholder="anurag")
    with col2:
        uploaded_file = st.file_uploader(
            "🎤 Upload clean voice sample (10-60 seconds)",
            type=['wav', 'mp3', 'm4a']
        )
    
    if st.button("🚀 Train My Model!", type="primary", use_container_width=True):
        if username and uploaded_file:
            # Save audio file
            audio_path = MODELS_DIR / f"{username}_voice.{uploaded_file.name.split('.')[-1]}"
            with open(audio_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Save model metadata
            save_user_model(username, audio_path)
            
            st.success(f"✅ **{username.title()}'s model trained successfully!** 🎉")
            st.balloons()
            st.rerun()
        else:
            st.error("❌ Enter username + upload audio file")

# TAB 2: Generate speech
with tab2:
    st.header("🎤 Generate Speech in Any Voice")
    
    # Show available models
    all_models = get_all_models()
    
    if all_models:
        st.success(f"✅ **{len(all_models)} voice models** available!")
        
        # Model selection
        model_names = [f"{m['username'].title()} ({m['created'][:10]})" for m in all_models]
        selected_idx = st.selectbox("🎙️ Select voice:", range(len(model_names)), format_func=lambda i: model_names[i])
        selected_model = all_models[selected_idx]
        
        # Preview original voice
        st.subheader("👂 Preview Original Voice")
        with open(selected_model["speaker_wav"], "rb") as f:
            st.audio(f.read(), format="audio/wav")
        
        # Text input
        st.subheader("✍️ Type text to generate")
        text = st.text_area(
            "Text to speak:",
            "Hello! This is my perfectly cloned voice. It sounds just like me!",
            height=120
        )
        
        col1, col2 = st.columns(2)
        with col1:
            language = st.selectbox("🌐 Language:", ["en", "es", "fr", "de", "hi", "pt"])
        with col2:
            temperature = st.slider("🎭 Emotion (0.5=neutral, 0.8=expressive)", 0.5, 0.8, 0.65)
        
        # Generate button
        if st.button("🎬 GENERATE PERFECT CLONE!", type="primary", use_container_width=True):
            with st.spinner("🎙️ Cloning voice... (30-60 seconds)"):
                output_path = f"output_{int(time.time())}.wav"
                
                tts.tts_to_file(
                    text=text,
                    speaker_wav=selected_model["speaker_wav"],
                    language=language,
                    file_path=output_path,
                    temperature=temperature,
                    speed=1.0
                )
                
                # Display result
                with open(output_path, "rb") as f:
                    audio_bytes = f.read()
                
                st.success("✅ **Perfect voice clone generated!** 🎉")
                st.audio(audio_bytes, format="audio/wav")
                
                # Download buttons
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="⬇️ Download WAV",
                        data=audio_bytes,
                        file_name=f"{selected_model['username']}_clone.wav",
                        mime="audio/wav"
                    )
                with col2:
                    # Convert to MP3 for smaller size
                    from pydub import AudioSegment
                    audio = AudioSegment.from_wav(output_path)
                    mp3_buffer = io.BytesIO()
                    audio.export(mp3_buffer, format="mp3")
                    mp3_bytes = mp3_buffer.getvalue()
                    st.download_button(
                        label="⬇️ Download MP3",
                        data=mp3_bytes,
                        file_name=f"{selected_model['username']}_clone.mp3",
                        mime="audio/mpeg"
                    )
    else:
        st.info("👆 **No voice models yet!** Go to 'Train Voice Model' tab to add the first voice!")
        st.markdown("### 📱 **Demo Flow:**")
        st.markdown("""
        1. **Anurag** uploads voice → "Anurag model" saved
        2. **Sakshi** uploads voice → "Sakshi model" saved  
        3. **Generate Speech** → Dropdown: `[Anurag | Sakshi]`
        4. Select "Anurag" → Type text → **Anurag's perfect voice!** 🎙️
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666'>
    <p>🎙️ Multi-User Perfect Voice Cloner | Made with ❤️ for college project</p>
    <p><small>Powered by XTTS-v2 • Open Source • Runs locally or on Streamlit Cloud</small></p>
</div>
""", unsafe_allow_html=True)
