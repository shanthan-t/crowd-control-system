import numpy as np
import librosa
import soundfile as sf
import os
import random

# Configuration
DATASET_PATH = "dataset"
CLASSES = ["panic", "cheering"]
NUM_SAMPLES = 20  # Samples per class
DURATION = 2.0  # Seconds
SR = 22050

def ensure_folders():
    for c in CLASSES:
        path = os.path.join(DATASET_PATH, c)
        os.makedirs(path, exist_ok=True)
        print(f"Created/Verified folder: {path}")

def generate_panic_sound(duration, sr):
    """
    Simulates panic sound: High pitched, fluctuating amplitude, siren-like or scream-like.
    Constructed using high frequency sine waves with modulation.
    """
    t = np.linspace(0, duration, int(sr * duration))
    # Base frequency for a "scream" (high pitch, e.g., 1000Hz - 2500Hz)
    base_freq = random.uniform(1200, 2000)
    
    # Frequency modulation (vibrato/unstable pitch)
    mod_freq = 10.0 # 10 Hz shake
    modulation = 200 * np.sin(2 * np.pi * mod_freq * t)
    
    # Carrier
    audio = 0.8 * np.sin(2 * np.pi * (base_freq + modulation) * t)
    
    # Add some harmonic distortion
    audio += 0.4 * np.sin(2 * np.pi * (base_freq * 1.5 + modulation) * t)
    
    # Apply sudden amplitude spikes (screaming bursts)
    envelope = np.ones_like(t)
    # Burst envelope
    for _ in range(3):
        start = random.randint(0, len(t) - int(sr*0.5))
        end = start + int(sr * 0.3)
        envelope[start:end] *= 2.5 # Spike volume
    
    audio = audio * envelope
    
    # Add a little white noise
    noise = np.random.normal(0, 0.05, len(audio))
    return audio + noise

def generate_cheering_sound(duration, sr):
    """
    Simulates cheering/normal crowd: Pink/White noise, lower spectral centroid, chaotic but spread spectrum.
    """
    length = int(sr * duration)
    # White noise base
    white_noise = np.random.normal(0, 0.5, length)
    
    # Cheering is often pink-ish noise (power falls with frequency)
    # Approximate pink noise with cumulative sum of white noise? No, that's brown.
    # Let's just use filtered white noise or superposition of many random tones.
    
    # Simple crowd roar: Bandpass filtered noise
    # But for raw generation, just regular white noise is drastically different from the pure sine of panic.
    # Let's make it a mix of many lower frequency tones and noise.
    
    noise = white_noise
    
    # Add some "voices" (random mid-frequency tones)
    t = np.linspace(0, duration, length)
    for _ in range(10):
        freq = random.uniform(200, 800) # Human voice range
        noise += 0.1 * np.sin(2 * np.pi * freq * t + random.uniform(0, 2*np.pi))
        
    return noise / np.max(np.abs(noise)) # Normalize

def generate_dataset():
    ensure_folders()
    print(f"Generating {NUM_SAMPLES} samples per class...")
    
    for label in CLASSES:
        for i in range(NUM_SAMPLES):
            if label == "panic":
                audio = generate_panic_sound(DURATION, SR)
            else:
                audio = generate_cheering_sound(DURATION, SR)
            
            # Normalize to prevent clipping
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            
            filename = os.path.join(DATASET_PATH, label, f"{label}_{i+1}.wav")
            sf.write(filename, audio, SR)
            
    print("Dataset generation complete.")

if __name__ == "__main__":
    generate_dataset()
