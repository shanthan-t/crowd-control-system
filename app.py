import streamlit as st
import numpy as np
import pyaudio
import librosa
import time
import os
import audio_engine  # Import our existing logic

# --- Page Config ---
st.set_page_config(
    page_title="Sentinel Audio Engine",
    page_icon="🔊",
    layout="wide",
)

# --- Header ---
st.title("🔊 Sentinel Audio Engine")
st.markdown("### Real-time Panic Sound Detection")

# --- Sidebar ---
st.sidebar.header("Options")
mode = st.sidebar.radio("Select Mode", ["Live Microphone", "Upload Audio File"])

RMS_THRESHOLD = st.sidebar.slider("RMS Threshold (Volume)", 0.0, 0.5, audio_engine.RMS_THRESHOLD)
CENTROID_THRESHOLD = st.sidebar.slider("Spectral Centroid Threshold (Pitch)", 0.0, 5000.0, audio_engine.SPECTRAL_CENTROID_THRESHOLD)

# Override engine thresholds with slider values
audio_engine.RMS_THRESHOLD = RMS_THRESHOLD
audio_engine.SPECTRAL_CENTROID_THRESHOLD = CENTROID_THRESHOLD

# --- Live Mode ---
if mode == "Live Microphone":
    st.subheader("🎤 Live Audio Monitor")
    
    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("Start Listening", type="primary")
    with col2:
        stop_btn = st.button("Stop Listening")

    if 'listening' not in st.session_state:
        st.session_state.listening = False

    if start_btn:
        st.session_state.listening = True
    if stop_btn:
        st.session_state.listening = False

    status_placeholder = st.empty()
    metric_placeholder = st.empty()
    
    if st.session_state.listening:
        p = pyaudio.PyAudio()
        try:
            stream = p.open(format=audio_engine.FORMAT,
                            channels=audio_engine.CHANNELS,
                            rate=audio_engine.RATE,
                            input=True,
                            frames_per_buffer=audio_engine.CHUNK_SIZE)
            
            # Create a buffer for visualization/analysis
            frames_per_analysis = int(audio_engine.RATE * audio_engine.RECORD_SECONDS_BUFFER)
            buffer = np.zeros(frames_per_analysis, dtype=np.float32)

            while st.session_state.listening:
                try:
                    data_bytes = stream.read(audio_engine.CHUNK_SIZE, exception_on_overflow=False)
                    data_np = np.frombuffer(data_bytes, dtype=np.float32)
                    
                    # Update buffer
                    buffer = np.roll(buffer, -audio_engine.CHUNK_SIZE)
                    buffer[-audio_engine.CHUNK_SIZE:] = data_np

                    # Analyze
                    is_panic, details = audio_engine.detect_panic(buffer, audio_engine.RATE)

                    # Update UI
                    if is_panic:
                        status_placeholder.error(f"🚨 **PANIC DETECTED** 🚨")
                    else:
                        status_placeholder.success(f"✅ Safe")

                    metric_placeholder.text(details)
                    
                    # Small sleep to yield to UI loop? (Streamlit updates on rerun usually, but loop works with placeholders)
                    # No sleep needed, read is blocking for chunk time
                    
                except IOError:
                    continue
                except Exception as e:
                    st.error(f"Error: {e}")
                    break
        except Exception as e:
            st.error(f"Could not open microphone: {e}")
        finally:
            if 'stream' in locals():
                stream.stop_stream()
                stream.close()
            p.terminate()

# --- File Mode ---
elif mode == "Upload Audio File":
    st.subheader("📁 File Analysis")
    
    uploaded_file = st.file_uploader("Upload a WAV or MP3 file", type=["wav", "mp3"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        if st.button("Analyze File"):
            with st.spinner("Analyzing..."):
                # Save temp file because librosa.load needs a path usually (or file-like object, but path is safer for ffmpeg)
                # Librosa can load file-like objects since 0.7, let's try passing the uploaded_file directly
                try:
                    y, sr = librosa.load(uploaded_file, sr=audio_engine.RATE)
                    
                    # Sliding window analysis logic reused here
                    window_size = int(audio_engine.RATE * 1.0)
                    step_size = int(audio_engine.RATE * 0.5)
                    
                    panic_timestamps = []
                    
                    progress_bar = st.progress(0)
                    total_steps = (len(y) - window_size) // step_size
                    
                    for idx, i in enumerate(range(0, len(y) - window_size, step_size)):
                        chunk = y[i : i+window_size]
                        is_panic, details = audio_engine.detect_panic(chunk, sr)
                        
                        if is_panic:
                            timestamp = i / sr
                            panic_timestamps.append((timestamp, details))
                        
                        if total_steps > 0:
                            progress_bar.progress(min((idx + 1) / total_steps, 1.0))

                    st.markdown("### Analysis Results")
                    if panic_timestamps:
                        st.error(f"⚠️ Panic detected in {len(panic_timestamps)} segments.")
                        for ts, det in panic_timestamps:
                            st.write(f"- **{ts:.1f}s**: {det}")
                    else:
                        st.success("✅ No panic sounds detected.")

                except Exception as e:
                    st.error(f"Error processing file: {e}")
