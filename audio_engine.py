import os
import sys
import time
import queue
import threading
import numpy as np
import pyaudio
import librosa

# --- Configuration ---
CHUNK_SIZE = 1024  # Number of frames per buffer
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 22050  # Sampling rate (Librosa default style)
RECORD_SECONDS_BUFFER = 1.0  # Analyze chunks of this duration

# --- Thresholds for "Panic" Detection (Heuristic Prototype) ---
# These values are arbitrary starting points and should be tuned.
# Panic is assumed to have high energy, high pitch (centroid), and sudden onset.
RMS_THRESHOLD = 0.05  # Minimum volume
SPECTRAL_CENTROID_THRESHOLD = 2000.0  # Frequency in Hz (Panic often high pitched)
# In a real model, we would use more sophisticated classifiers.

def install_dependencies():
    print("Dependencies required:")
    print("pip install pyaudio librosa numpy")
    print("Note: PyAudio may require system-level audio development headers (e.g., portaudio19-dev on Linux).")

def detect_panic(y, sr):
    """
    Analyzes audio data and determines if it indicates 'Panic'.
    Returns: (is_panic, details_string)
    """
    try:
        # 1. RMS Energy (Loudness)
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)

        # 2. Spectral Centroid (Brightness/Pitch)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        cent_mean = np.mean(cent)

        # Simple Decision Logic
        is_loud = rms_mean > RMS_THRESHOLD
        is_high_pitch = cent_mean > SPECTRAL_CENTROID_THRESHOLD
        
        # Panic detected if BOTH loud and high-pitched (simplified heuristic)
        is_panic = is_loud and is_high_pitch

        status_text = "SAFE"
        if is_panic:
            status_text = "PANIC DETECTED"

        details = f"RMS: {rms_mean:.4f} | Centroid: {cent_mean:.0f} Hz | Result: {status_text}"
        return is_panic, details

    except Exception as e:
        return False, f"Error in detection: {e}"

def analyze_audio_chunk(audio_chunk, sr):
    """
    Wrapper for detection on a chunk of audio.
    audio_chunk: numpy array of float32
    """
    # Ensure raw bytes are converted to numpy if not already (PyAudio gives bytes)
    if isinstance(audio_chunk, bytes):
        audio_data = np.frombuffer(audio_chunk, dtype=np.float32)
    else:
        audio_data = audio_chunk

    # Remove NaNs or Infs
    audio_data = np.nan_to_num(audio_data)
    
    return detect_panic(audio_data, sr)

# --- Live Audio Mode ---

def record_live_audio():
    """
    Captures live audio from default microphone and processes it in real-time.
    """
    p = pyaudio.PyAudio()

    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK_SIZE)
    except IOError as e:
        print(f"Error opening audio stream: {e}")
        print("Please check your microphone connection.")
        return

    print("\n--- LIVE AUDIO DETECTION MODE ---")
    print("Listening... Press Ctrl+C to stop.")
    
    # Buffer to hold enough audio for analysis (e.g., 1 second)
    frames_per_analysis = int(RATE * RECORD_SECONDS_BUFFER)
    buffer = np.zeros(frames_per_analysis, dtype=np.float32)
    
    try:
        while True:
            # Read chunk
            try:
                data_bytes = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                data_np = np.frombuffer(data_bytes, dtype=np.float32)
            except IOError as e:
                # print(f"Audio overflow/error: {e}")
                continue

            # Scroll buffer and append new data
            buffer = np.roll(buffer, -CHUNK_SIZE)
            buffer[-CHUNK_SIZE:] = data_np

            # Analyze the full buffer
            is_panic, details = analyze_audio_chunk(buffer, RATE)

            # Output
            if is_panic:
                print(f"\033[91m{details}\033[0m") # Red text for panic
            else:
                sys.stdout.write(f"\r{details}") # Overwrite line for safe
                sys.stdout.flush()

    except KeyboardInterrupt:
        print("\nStopping live detection...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

# --- File Upload Mode ---

def process_file(filepath):
    """
    Loads an audio file and processes it. 
    Simulates real-time by breaking it into chunks or analyzes it entirely?
    User asked for "Upload Audio File Mode... Process...". 
    We will analyze the whole file to give an overall verdict, 
    but for a 'prototype' showing real-time capability, scanning it in chunks is cooler.
    Let's just analyze the whole file for simplicity and accuracy unless 
    it's very long, but the prompt implies simple detection.
    Let's do a sliding window analysis to pinpoint panic segments.
    """
    if not os.path.isfile(filepath):
        print(f"File not found: {filepath}")
        return

    print(f"Loading {filepath} ...")
    try:
        y, sr = librosa.load(filepath, sr=RATE)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print("Analyzing file...")
    
    # Sliding window analysis
    window_size = int(RATE * 1.0) # 1 second window
    step_size = int(RATE * 0.5)   # 0.5 second overlap
    
    panic_events = 0
    total_windows = 0

    for i in range(0, len(y) - window_size, step_size):
        chunk = y[i : i+window_size]
        is_panic, details = detect_panic(chunk, sr)
        timestamp = i / sr
        
        status = "PANIC" if is_panic else "SAFE"
        if is_panic:
            print(f"[{timestamp:.1f}s] {status} - {details}")
            panic_events += 1
        total_windows += 1

    print("\n--- Summary ---")
    if panic_events > 0:
        print(f"RESULT: PANIC DETECTED ({panic_events} segments found)")
    else:
        print("RESULT: SAFE (No panic sounds detected)")

# --- Main Menu ---

def main():
    while True:
        print("\n========================================")
        print("   SENTINEL AUDIO ENGINE PROTOTYPE")
        print("========================================")
        print("1. Live Microphone Detection")
        print("2. Upload/Analyze Audio File")
        print("3. Exit")
        
        choice = input("Select an option (1-3): ").strip()

        if choice == '1':
            record_live_audio()
        elif choice == '2':
            path = input("Enter path to audio file: ").strip()
            # Remove quotes if user dragged and dropped file
            path = path.replace("'", "").replace('"', "")
            process_file(path)
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    main()
