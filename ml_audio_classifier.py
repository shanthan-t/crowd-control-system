import os
import sys
import glob
import numpy as np
import librosa
import pyaudio
import joblib
import warnings
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Suppress librosa warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
DATASET_PATH = "dataset"
MODEL_PATH = "audio_classifier_model.pkl"
SCALER_PATH = "scaler.pkl"
CLASSES = ["panic", "cheering"]
SAMPLE_RATE = 22050
DURATION = 2.0  # Training sample and Live chunk duration
N_MFCC = 13

# --- Feature Extraction ---
def extract_features(y, sr):
    """
    Extracts a feature vector from an audio time series.
    Features: MFCC(mean, var), RMS, ZCR, Centroid, Bandwidth, Flatness, Flux(contrast?).
    
    Returns: 1D numpy array of features.
    """
    if len(y) == 0:
        return None

    # 1. MFCCs (Timbre)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_var = np.var(mfccs, axis=1) # Variance is useful for "chaotic" sounds like screams

    # 2. RMS Energy (Loudness)
    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)

    # 3. Zero Crossing Rate (Noisiness/Pitch approx)
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)

    # 4. Spectral Centroid (Brightness)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    cent_mean = np.mean(centroid)

    # 5. Spectral Bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    bw_mean = np.mean(bandwidth)

    # 6. Spectral Flatness (Tonality vs Noise) (Screams = Tonal, Crowd = Noise)
    flatness = librosa.feature.spectral_flatness(y=y)
    flat_mean = np.mean(flatness)

    # Concatenate all features
    features = np.hstack([mfcc_mean, mfcc_var, rms_mean, zcr_mean, cent_mean, bw_mean, flat_mean])
    return features

# --- Dataset Loading & Training ---
def load_dataset():
    print("Loading dataset...")
    X = []
    y = []
    
    for label in CLASSES:
        folder = os.path.join(DATASET_PATH, label)
        files = glob.glob(os.path.join(folder, "*.wav"))
        
        print(f"Processing {len(files)} files for class '{label}'...")
        for file in files:
            try:
                audio, sr = librosa.load(file, sr=SAMPLE_RATE, duration=DURATION)
                # Padding if too short
                if len(audio) < SAMPLE_RATE * DURATION:
                    audio = librosa.util.fix_length(audio, size=int(SAMPLE_RATE*DURATION))
                
                feat = extract_features(audio, sr)
                if feat is not None:
                    X.append(feat)
                    y.append(label)
            except Exception as e:
                print(f"Error loading {file}: {e}")

    return np.array(X), np.array(y)

def train_model():
    X, y = load_dataset()
    
    if len(X) == 0:
        print("Error: No data found. Run generate_dataset.py first!")
        return None, None

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train SVM (Good for small datasets/high dim)
    print("Training SVM Classifier...")
    model = SVC(kernel='rbf', probability=True)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {acc*100:.2f}%")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    return model, scaler

# --- Inference ---
def load_model_if_exists():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        print("Loading existing model...")
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    else:
        print("Model not found. Training new model...")
        return train_model()

def predict(model, scaler, audio, sr):
    features = extract_features(audio, sr)
    features_scaled = scaler.transform([features])
    
    # Probabilities
    probs = model.predict_proba(features_scaled)[0]
    class_idx = np.argmax(probs)
    label = model.classes_[class_idx]
    confidence = probs[class_idx]
    
    return label, confidence

# --- Live Mode ---
def live_mode(model, scaler):
    p = pyaudio.PyAudio()
    
    CHUNK_SIZE = 1024
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    
    # Buffer length = Duration needed for feature extraction
    FRAMES_PER_BUFFER = int(SAMPLE_RATE * DURATION)
    
    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=SAMPLE_RATE,
                        input=True,
                        frames_per_buffer=CHUNK_SIZE)
    except IOError as e:
        print(f"Microphone error: {e}")
        return

    print("\n--- ML LIVE DETECTION (Press Ctrl+C to Stop) ---")
    
    buffer = np.zeros(FRAMES_PER_BUFFER, dtype=np.float32)
    
    # False Alarm Smoothing
    history = []
    SMOOTHING_WINDOW = 3 # Consecutive detections needed
    
    try:
        while True:
            # Read smaller chunks to keep loop responsive, but fill buffer
            # Actually, for simplicity, let's read the full duration or handle a rolling buffer
            # Reading 2 seconds at once is simple but high latency.
            # Use rolling buffer.
            
            data_bytes = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            data_np = np.frombuffer(data_bytes, dtype=np.float32)
            
            # Roll buffer
            buffer = np.roll(buffer, -CHUNK_SIZE)
            buffer[-CHUNK_SIZE:] = data_np
            
            # Predict every chunk? Can be CPU intensive.
            # Let's predict only when we have filled 'enough' new data? 
            # For now, just predict every loop (approx 20 times a sec).
            # To reduce load, maybe skip frames or just let it run.
            
            label, conf = predict(model, scaler, buffer, SAMPLE_RATE)
            
            # Smoothing Logic
            history.append(label)
            if len(history) > SMOOTHING_WINDOW:
                history.pop(0)
            
            # If all last N are PANIC, trigger panic
            if history.count("panic") == SMOOTHING_WINDOW:
                final_status = "PANIC DETECTED"
                color_code = "\033[91m" # Red
            else:
                final_status = "CHEERING / SAFE"
                color_code = "\033[92m" # Green
            
            sys.stdout.write(f"\r{color_code}STATUS: {final_status} (Conf: {conf:.2f}) [Raw: {label}]\033[0m")
            sys.stdout.flush()
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

# --- File Mode ---
def file_mode(model, scaler):
    path = input("Enter path to audio file: ").strip().replace("'", "").replace('"', "")
    if not os.path.exists(path):
        print("File not found.")
        return
        
    print(f"Analyzing {path}...")
    try:
        # Analyze whole file? Or chunks? 
        # ML model is trained on 2.0s clips.
        # We should split file into 2.0s windows.
        y, sr = librosa.load(path, sr=SAMPLE_RATE)
        
        window_samples = int(DURATION * SAMPLE_RATE)
        stride = int(window_samples / 2) # 50% overlap
        
        panic_count = 0
        total_windows = 0
        
        print("\n--- Timeline ---")
        for i in range(0, len(y) - window_samples, stride):
            chunk = y[i : i+window_samples]
            label, conf = predict(model, scaler, chunk, sr)
            timestamp = i / sr
            
            if label == "panic":
                print(f"[{timestamp:.1f}s] PANIC DETECTED (Conf: {conf:.2f})")
                panic_count += 1
            total_windows += 1
            
        print("\n--- Summary ---")
        if panic_count > 0:
            print(f"RESULT: PANIC sounds detected in {panic_count} segments.")
        else:
            print("RESULT: SAFE (Mostly Cheering/Normal)")
            
    except Exception as e:
        print(f"Error: {e}")

# --- Main ---
def main():
    print("Initializing Machine Learning Audio Classifier...")
    model, scaler = load_model_if_exists()
    
    if model is None:
        return

    while True:
        print("\n=== SENTINEL ML CLASSIFIER ===")
        print("1. Live Audio Mode")
        print("2. Analyze Audio File")
        print("3. Retrain Model (Force)")
        print("4. Exit")
        
        choice = input("Select: ").strip()
        
        if choice == '1':
            live_mode(model, scaler)
        elif choice == '2':
            file_mode(model, scaler)
        elif choice == '3':
            model, scaler = train_model()
        elif choice == '4':
            break

if __name__ == "__main__":
    main()
