import cv2
import numpy as np
import pyaudio
import librosa
import threading
import time
import sys
import queue
from ultralytics import YOLO

# --- Configuration ---
# Vision
YOLO_MODEL = "yolov8n-pose.pt"  # Nano Pose model for speed
CROWD_DENSITY_THRESHOLD_HIGH = 5  # People count for High Density (Prototype value)
CROWD_DENSITY_THRESHOLD_MEDIUM = 3

# Audio
CHUNK_SIZE = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 22050
AUDIO_BUFFER_SECONDS = 1.0  # Analyze 1-second chunks
# Heuristic Thresholds (tuned for prototype)
RMS_THRESHOLD = 0.05
SPECTRAL_CENTROID_THRESHOLD = 2000.0
# Panic Persistence: Number of consecutive panic frames to trigger "Global Panic"
PANIC_PERSISTENCE = 3 

# Application State
class SentinelState:
    def __init__(self):
        self.running = True
        
        # Shared Data
        self.crowd_count = 0
        self.crowd_status = "LOW"  # LOW, MEDIUM, HIGH
        
        self.audio_status = "NORMAL" # NORMAL, PANIC
        self.audio_status_raw = "NORMAL" # Instantaneous status
        self.audio_features = {} # For debugging/display
        self.panic_counter = 0 # Temporal smoothing

        self.risk_level = "LOW" # LOW, MEDIUM, HIGH
        
        self.frame = None # Latest video frame
        self.lock = threading.Lock()

state = SentinelState()

# --- Audio Engine (Threaded) ---
def detect_panic_audio(y, sr):
    """
    Analyzes audio chunk for panic features.
    Returns: (is_panic, features_dict)
    """
    try:
        # Features
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)
        
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        cent_mean = np.mean(centroid)
        
        # Additional features requested
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        bw_mean = np.mean(bandwidth)
        
        flatness = librosa.feature.spectral_flatness(y=y)
        flat_mean = np.mean(flatness)
        
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)

        # Heuristic Logic (Fast & Robust for prototype without training data)
        # Panic = Loud (RMS) AND High Pitch (Centroid)
        is_panic = (rms_mean > RMS_THRESHOLD) and (cent_mean > SPECTRAL_CENTROID_THRESHOLD)
        
        features = {
            "RMS": rms_mean,
            "Centroid": cent_mean,
            "Bandwidth": bw_mean,
            "ZCR": zcr_mean
        }
        return is_panic, features

    except Exception as e:
        print(f"Audio Analysis Error: {e}")
        return False, {}

def process_audio_live():
    print("[Audio] Initializing Live Mic...")
    p = pyaudio.PyAudio()
    buffer_len = int(RATE * AUDIO_BUFFER_SECONDS)
    audio_buffer = np.zeros(buffer_len, dtype=np.float32)
    
    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK_SIZE)
        
        while state.running:
            try:
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                chunk = np.frombuffer(data, dtype=np.float32)
                
                # Roll buffer
                audio_buffer = np.roll(audio_buffer, -CHUNK_SIZE)
                audio_buffer[-CHUNK_SIZE:] = chunk
                
                # Analyze (Frequency: every chunk might be too fast, let's limit)
                # But read is blocking, so it's fine.
                
                is_panic, feats = detect_panic_audio(audio_buffer, RATE)
                
                # Update State
                with state.lock:
                    state.audio_features = feats
                    state.audio_status_raw = "PANIC" if is_panic else "NORMAL"
                    
                    # Temporal Smoothing
                    if is_panic:
                        state.panic_counter += 1
                        if state.panic_counter >= PANIC_PERSISTENCE:
                            state.audio_status = "PANIC"
                            # Clamp counter to avoid overflow/hysteresis issues
                            state.panic_counter = PANIC_PERSISTENCE + 2 
                    else:
                        state.panic_counter -= 1
                        if state.panic_counter <= 0:
                            state.audio_status = "NORMAL"
                            state.panic_counter = 0

            except Exception as e:
                # print(f"Audio Stream Error: {e}")
                pass
                
    except Exception as e:
        print(f"[Audio] Error: {e}")
    finally:
        if 'stream' in locals():
            stream.stop_stream()
            stream.close()
        p.terminate()
        print("[Audio] Stopped.")

def process_audio_file(filepath):
    print(f"[Audio] Analyzing file: {filepath}")
    y, sr = librosa.load(filepath, sr=RATE)
    
    # Simulate playback/streaming
    chunk_samples = CHUNK_SIZE
    total_samples = len(y)
    cursor = 0
    
    window_size = int(RATE * AUDIO_BUFFER_SECONDS)
    
    while state.running and cursor < total_samples:
        start_time = time.time()
        
        # Extract window for analysis (simulating buffer state at this time)
        # For simplicity, let's just slide a window.
        if cursor + window_size < total_samples:
             current_window = y[cursor : cursor + window_size]
        else:
             current_window = y[cursor:]
             if len(current_window) < 100: break

        is_panic, feats = detect_panic_audio(current_window, sr)
        
        with state.lock:
            state.audio_features = feats
            state.audio_status_raw = "PANIC" if is_panic else "NORMAL"
             # Simplified smoothing for file mode
            state.audio_status = state.audio_status_raw 

        # Advance cursor
        cursor += chunk_samples
        
        # Sync with real-time (approx)
        process_time = time.time() - start_time
        sleep_time = (chunk_samples / RATE) - process_time
        if sleep_time > 0:
            time.sleep(sleep_time)

# --- Vision Engine ---

def process_video_generic(source):
    print(f"[Vision] Loading YOLOv8 model ({YOLO_MODEL})...")
    model = YOLO(YOLO_MODEL)
    print(f"[Vision] Opening source: {source}")
    
    # Check if source is digit (webcam index)
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"[Vision] Error: Could not open video source {source}")
        state.running = False
        return

    while state.running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[Vision] End of stream.")
            break
        
        # YOLO Inference
        results = model(frame, verbose=False, classes=[0]) # Class 0 = Person
        
        person_count = 0
        annotated_frame = frame.copy()
        
        for r in results:
            # Helper to draw boxes
            annotated_frame = r.plot()
            person_count = len(r.boxes)
            
        # Update State
        with state.lock:
            state.crowd_count = person_count
            state.frame = annotated_frame
            
            # Density Logic
            if person_count >= CROWD_DENSITY_THRESHOLD_HIGH:
                state.crowd_status = "HIGH"
            elif person_count >= CROWD_DENSITY_THRESHOLD_MEDIUM:
                state.crowd_status = "MEDIUM"
            else:
                state.crowd_status = "LOW"
            
            # --- Fusion Link ---
            fusion_logic()

        # Display is handled in main thread or here?
        # OpenCV imshow must run in main thread usually on some OS (Mac). 
        # But commonly in scripts, we run loop in main and input in threads.
        # User accepted multithreading "so camera and audio run simultaneously".
        # Let's keep imshow here for simplicity if main thread is blocked by menu, 
        # OR run this loop in main thread.
        # Decision: Run Vision in Main Thread (it controls the window), Audio in Background Thread.
        
        draw_hud(state.frame)
        cv2.imshow("Sentinel Integrated System", state.frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            state.running = False
            break

    cap.release()
    cv2.destroyAllWindows()
    state.running = False

def fusion_logic():
    """
    Combines Vision and Audio data to determine Risk.
    Called inside the update lock.
    """
    audio_p = state.audio_status == "PANIC"
    crowd_h = state.crowd_status == "HIGH"
    crowd_m = state.crowd_status == "MEDIUM"
    
    if audio_p and crowd_h:
        state.risk_level = "HIGH"
    elif audio_p:
        # Panic sound but low crowd -> maybe isolated incident
        state.risk_level = "MEDIUM" 
    elif crowd_h:
         # High crowd but silent -> Potential risk or monitoring needed
         state.risk_level = "MEDIUM"
    else:
        state.risk_level = "LOW"

def draw_hud(frame):
    """
    Draws Overlay with People Count, Audio Status, and Risk Level.
    """
    # Background Box
    cv2.rectangle(frame, (10, 10), (350, 160), (0, 0, 0), -1)
    
    # Colors
    risk_color = (0, 255, 0) # Green
    if state.risk_level == "MEDIUM": risk_color = (0, 255, 255) # Yellow
    if state.risk_level == "HIGH": risk_color = (0, 0, 255) # Red
    
    audio_color = (0, 0, 255) if state.audio_status == "PANIC" else (0, 255, 0)
    
    # Text
    cv2.putText(frame, "SENTINEL SYSTEM", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Vision Data
    cv2.putText(frame, f"People: {state.crowd_count}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Density: {state.crowd_status}", (150, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # Audio Data
    cv2.putText(frame, f"Audio: {state.audio_status}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, audio_color, 2)
    
    # Risk Level
    cv2.putText(frame, f"RISK: {state.risk_level}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, risk_color, 3)

# --- Main Menu ---
def main():
    print("==========================================")
    print("   SENTINEL INTEGRATED PROTOTYPE (V+A)    ")
    print("==========================================")
    print("1. Live Camera + Live Microphone")
    print("2. Video File + Audio File")
    print("3. Exit")
    
    choice = input("Select Mode: ").strip()
    
    if choice == '1':
        # Start Audio Thread
        t_audio = threading.Thread(target=process_audio_live)
        t_audio.start()
        
        # Start Vision (Main Thread)
        process_video_generic(0) # 0 for Webcam
        
        t_audio.join()

    elif choice == '2':
        v_path = input("Path to Video: ").strip().replace("'", "").replace('"', "")
        a_path = input("Path to Audio: ").strip().replace("'", "").replace('"', "")
        
        t_audio = threading.Thread(target=process_audio_file, args=(a_path,))
        t_audio.start()
        
        process_video_generic(v_path)
        
        t_audio.join()
        
    else:
        print("Exiting.")

if __name__ == "__main__":
    main()
