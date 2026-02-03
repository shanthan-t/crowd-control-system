import streamlit as st
import cv2
import numpy as np
import pyaudio
import librosa
import threading
import time
from ultralytics import YOLO
import sentinel_hub  # Integration

# --- Configuration ---
# YOLO_MODEL = "yolov8n.pt" # dynamic now
CROWD_DENSITY_HIGH = 5
AUDIO_RATE = 22050
AUDIO_CHUNK = 1024

# --- State Management ---
if 'audio_status' not in st.session_state:
    st.session_state.audio_status = "NORMAL"
if 'running' not in st.session_state:
    st.session_state.running = False

# --- Audio Thread ---
def audio_listener():
    """
    Background thread to listen to microphone and update session state.
    Note: Streamlit session state is not thread-safe in the usual way, 
    so we use a global or a mutable object if we want to share data? 
    Actually, threads spawned by Streamlit re-run the script. 
    It's tricky.
    
    Better approach for Streamlit Live Loop:
    Run the logic INSIDE the main loop frame-by-frame.
    Audio needs to be non-blocking.
    
    We will use PyAudio non-blocking callback or just read small chunks in the loop.
    For this prototype, let's try to read audio in the loop.
    """
    pass 

# --- Helper Logic ---
def analyze_audio_chunk(stream):
    try:
        # Read without blocking too long?
        if stream.get_read_available() >= AUDIO_CHUNK:
            data = stream.read(AUDIO_CHUNK, exception_on_overflow=False)
            y = np.frombuffer(data, dtype=np.float32)
            
            rms = np.mean(librosa.feature.rms(y=y))
            cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=AUDIO_RATE))
            
            # Simple Thresholds
            if rms > 0.05 and cent > 2000:
                return "PANIC"
    except:
        pass
    return "NORMAL"

def get_risk_level(audio_status, person_count):
    if audio_status == "PANIC" and person_count >= CROWD_DENSITY_HIGH:
        return "HIGH", "red"
    elif audio_status == "PANIC" or person_count >= CROWD_DENSITY_HIGH:
        return "MEDIUM", "orange"
    else:
        return "LOW", "green"

# --- Main App ---
def main():
    st.set_page_config(page_title="Sentinel Integrated System", layout="wide")
    
    st.title("🛡️ Sentinel Integrated System")
    st.markdown("**Real-time Fusion of Computer Vision & Audio Analysis**")

    # Sidebar
    st.sidebar.header("Settings")
    mode = st.sidebar.radio("Input Source", ["Live Webcam & Mic", "Video & Audio File"])
    
    st.sidebar.divider()
    model_type = st.sidebar.selectbox(
        "YOLO Pose Model", 
        ["Nano Pose (Fast)", "Small Pose (Balanced)", "Medium Pose (Accuracy)", "Large Pose (High Acc)", "Huge Pose (Best Acc)"], 
        index=0
    )
    conf_thresh = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.25, 0.05)
    
    st.sidebar.markdown("### Advanced Accuracy")
    img_size = st.sidebar.select_slider("Inference Resolution (px)", options=[640, 960, 1280], value=640)
    iou_thresh = st.sidebar.slider("NMS IOU Threshold", 0.1, 1.0, 0.45, 0.05, help="Lower values reduce overlapping boxes.")
    
    model_map = {
        "Nano Pose (Fast)": "yolov8n-pose.pt",
        "Small Pose (Balanced)": "yolov8s-pose.pt",
        "Medium Pose (Accuracy)": "yolov8m-pose.pt",
        "Large Pose (High Acc)": "yolov8l-pose.pt",
        "Huge Pose (Best Acc)": "yolov8x-pose.pt"
    }
    selected_model = model_map[model_type]
    
    # Model Loading (Cached)
    @st.cache_resource
    def load_model(model_name):
        return YOLO(model_name)
    
    model = load_model(selected_model)

    # Hub Initialization (Singleton)
    hub = sentinel_hub.get_hub()

    if mode == "Live Webcam & Mic":
        start_button = st.button("Start System", type="primary")
        stop_button = st.button("Stop")
        
        if start_button:
            st.session_state.running = True
        if stop_button:
            st.session_state.running = False

        if st.session_state.running:
            # placeholders
            video_placeholder = st.empty()
            col1, col2, col3, col4 = st.columns(4)
            metric_people = col1.empty()
            metric_skeleton = col2.empty()
            metric_audio = col3.empty()
            metric_risk = col4.empty()
            
            # Init Audio
            p = pyaudio.PyAudio()
            try:
                stream = p.open(format=pyaudio.paFloat32,
                                channels=1,
                                rate=AUDIO_RATE,
                                input=True,
                                frames_per_buffer=AUDIO_CHUNK)
                
                # Init Video
                cap = cv2.VideoCapture(0)
                
                panic_counter = 0 # For persistence
                
                while st.session_state.running:
                    # 1. Video Capture
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to capture video.")
                        break
                        
                    # 2. Vision Inference
                    results = model(frame, verbose=False, classes=[0], conf=conf_thresh, imgsz=img_size, iou=iou_thresh)
                    person_count = 0
                    annotated_frame = frame
                    skeleton_detected = "NO"
                    
                    for r in results:
                        annotated_frame = r.plot()
                        person_count = len(r.boxes)
                        if r.keypoints is not None and len(r.keypoints) > 0:
                            skeleton_detected = "YES"
                    
                    # 3. Audio Analysis
                    # Read all available chunks to prevent accumulation
                    status_audio = "NORMAL"
                    try:
                        while stream.get_read_available() > AUDIO_CHUNK:
                             data = stream.read(AUDIO_CHUNK, exception_on_overflow=False)
                             y = np.frombuffer(data, dtype=np.float32)
                             rms = np.mean(librosa.feature.rms(y=y))
                             cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=AUDIO_RATE))
                             
                             if rms > 0.05 and cent > 2000:
                                 status_audio = "PANIC"
                    except:
                        pass
                    
                    # Smoothing
                    if status_audio == "PANIC":
                        panic_counter += 1
                    else:
                        panic_counter = max(0, panic_counter - 1)
                        
                    final_audio_status = "PANIC" if panic_counter > 2 else "NORMAL"

                    # 4. Fusion Logic
                    risk, color = get_risk_level(final_audio_status, person_count)

                    # --- HUB LOGGING ---
                    # Log every N frames to avoid flooding? Or every frame?
                    # For prototype, every frame is fine, or maybe check change.
                    # Let's log every frame for real-time socket updates.
                    hub.log_result(person_count, final_audio_status, risk)
                    hub.broadcast_frame(annotated_frame) # Broadcast Video to React
                    # -------------------

                    # 5. Display Updates
                    # Convert BGR to RGB for Streamlit
                    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    
                    # Draw Risk Overlay on Frame (optional, or just use metrics)
                    # Let's use metrics for clarity
                    
                    video_placeholder.image(frame_rgb,  channels="RGB", use_container_width=True)
                    
                    metric_people.metric("People Count", person_count)
                    metric_skeleton.metric("Skeleton Detected", skeleton_detected)
                    metric_audio.metric("Audio Status", final_audio_status,delta_color="inverse")
                    metric_risk.markdown(f"### Risk Level: :{color}[{risk}]")
                    
                    # Loop speed control? YOLO is the bottleneck usually.
                
                cap.release()
                stream.stop_stream()
                stream.close()
                p.terminate()

            except Exception as e:
                st.error(f"Error: {e}")

    elif mode == "Video & Audio File":
        st.subheader("📂 File Analysis")
        
        col1, col2 = st.columns(2)
        v_file = col1.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
        a_file = col2.file_uploader("Upload Audio (Optional)", type=['wav', 'mp3'])
        
        if v_file is not None:
            # Save files
            v_path = f"temp_uploads/{v_file.name}"
            with open(v_path, "wb") as f:
                f.write(v_file.getbuffer())
            
            a_path = None
            if a_file:
                a_path = f"temp_uploads/{a_file.name}"
                with open(a_path, "wb") as f:
                    f.write(a_file.getbuffer())

            if st.button("Analyze Uploaded Files"):
                
                # placeholders
                video_placeholder = st.empty()
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                metric_people = m_col1.empty()
                metric_skeleton = m_col2.empty() 
                metric_audio = m_col3.empty()
                metric_risk = m_col4.empty()
                
                # Load Audio
                y_audio = None
                sr_audio = AUDIO_RATE
                if a_path:
                    with st.spinner("Loading audio..."):
                        y_audio, sr_audio = librosa.load(a_path, sr=AUDIO_RATE)

                # Open Video
                cap = cv2.VideoCapture(v_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0: fps = 30 # Fallback
                
                frame_idx = 0
                panic_counter = 0

                st.info(f"Processing... FPS: {fps:.2f}")

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # 1. Vision Inference
                    results = model(frame, verbose=False, classes=[0], conf=conf_thresh, imgsz=img_size, iou=iou_thresh)
                    person_count = 0
                    annotated_frame = frame
                    skeleton_detected = "NO"
                    
                    for r in results:
                        annotated_frame = r.plot()
                        person_count = len(r.boxes)
                        if r.keypoints is not None and len(r.keypoints) > 0:
                             skeleton_detected = "YES"
                    
                    # 2. Audio Analysis (Sync)
                    status_audio = "NORMAL"
                    if y_audio is not None:
                        # Map frame to audio time
                        current_time = frame_idx / fps
                        sample_idx = int(current_time * sr_audio)
                        
                        # Extract 1 second window centered or following current time?
                        # Let's take [current, current+1s]
                        window_size = int(sr_audio * 1.0)
                        
                        if sample_idx < len(y_audio):
                            end_idx = min(sample_idx + window_size, len(y_audio))
                            chunk = y_audio[sample_idx : end_idx]
                            
                            if len(chunk) > window_size // 2: # Only analyze if enough data
                                rms = np.mean(librosa.feature.rms(y=chunk))
                                cent = np.mean(librosa.feature.spectral_centroid(y=chunk, sr=sr_audio))
                                if rms > 0.05 and cent > 2000:
                                    status_audio = "PANIC"
                    
                    # Smoothing
                    if status_audio == "PANIC":
                        panic_counter += 1
                    else:
                        panic_counter = max(0, panic_counter - 1)
                    
                    # File mode persistence might need tuning, keep it simpler 
                    final_audio_status = "PANIC" if panic_counter > 2 else "NORMAL"

                    # 3. Fusion
                    risk, color = get_risk_level(final_audio_status, person_count)

                    # --- HUB LOGGING ---
                    hub.log_result(person_count, final_audio_status, risk)
                    hub.broadcast_frame(annotated_frame) # Broadcast Video
                    # -------------------

                    # 4. Display
                    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                    
                    metric_people.metric("People Count", person_count)
                    metric_skeleton.metric("Skeleton Detected", skeleton_detected)
                    metric_audio.metric("Audio Status", final_audio_status, delta_color="inverse")
                    metric_risk.markdown(f"### Risk Level: :{color}[{risk}]")
                    
                    frame_idx += 1
                    # Optional: Sleep to match roughly real-time viewing?
                    # Without sleep, it runs as fast as CPU processes frames.
                
                cap.release()
                st.success("Analysis Complete.")

if __name__ == "__main__":
    main()
