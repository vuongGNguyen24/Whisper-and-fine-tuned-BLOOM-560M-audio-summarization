import streamlit as st
from st_audiorec import st_audiorec
import requests
import io
MIN_LEN = 100

def summarize(transcription):
    if not transcription or len(transcription.split()) < MIN_LEN:
        st.write("Your transcript is too short to summarize")
        return
    with st.spinner("Summarizing..."):
        summary_response = requests.post("http://localhost:8000/summarize", json={"text": transcription})
        if summary_response.status_code == 200:
            summary = summary_response.json().get("summary", "")
            st.subheader("Summary")
            st.write(summary)
        else:
            st.error("Summarization failed!")

st.title("Speech Summarizer with whisper and fine-tuned BLOOM 560M")

def transcript(files):
    with st.spinner("Trancripting"):
        response = requests.post("http://localhost:8000/transcribe", files=files)
        if response.status_code == 200:
            transcription = response.json().get("text", "")
            st.subheader("Transcription")
            st.write(transcription)
            summarize(transcription)
        else:
            st.error("Transcription failed!")
# # --- Record audio from microphone ---
# wav_audio_data = st_audiorec()

# # --- Playback recorded audio ---
# if wav_audio_data is not None:
#     st.audio(wav_audio_data, format='audio/wav')
    
#     if st.button("📝 Summarize"):
#         st.info("Sending audio to backend... Please wait ⏳")

#         # Send recorded audio to backend for transcription
#         files = {"file": ("recorded.wav", wav_audio_data, "audio/wav")}
#         transcript(files)

# # --- Optional: Upload audio file instead of mic ---
# st.markdown("---")
uploaded_file = st.file_uploader("upload an audio file (wav/mp3/flac)")

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    
    if st.button("📄 Summarize Uploaded File"):
        st.info("Processing uploaded file...")
        files = {"file": uploaded_file}
        transcript(files)