import cv2
import numpy as np
import time
import mediapipe as mp

def test_perf():
    # Warmup
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=10,
        refine_landmarks=True,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4
    )
    
    # Dummy 640x360 frame
    frame = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
    
    # Warmup runs
    for _ in range(5):
        face_mesh.process(frame)
        
    runs = 50
    times = []
    
    for _ in range(runs):
        start = time.perf_counter()
        res = face_mesh.process(frame)
        end = time.perf_counter()
        times.append((end - start)*1000)
        
    avg_time = sum(times) / runs
    max_time = max(times)
    min_time = min(times)
    
    print(f"MediaPipe Face Mesh Performance (640x360):")
    print(f"  Avg: {avg_time:.2f} ms")
    print(f"  Min: {min_time:.2f} ms")
    print(f"  Max: {max_time:.2f} ms")
    
if __name__ == "__main__":
    test_perf()
