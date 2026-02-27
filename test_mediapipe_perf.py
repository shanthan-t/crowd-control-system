import cv2
import numpy as np
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def test_perf():
    # Warmup
    base_options = python.BaseOptions(model_asset_path='user_dashboard/blaze_face_short_range.tflite')
    options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.4)
    face_detector = vision.FaceDetector.create_from_options(options)
    
    # Dummy 640x360 frame
    frame = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    # Warmup runs
    for _ in range(5):
        face_detector.detect(mp_image)
        
    runs = 100
    times = []
    
    for _ in range(runs):
        start = time.perf_counter()
        res = face_detector.detect(mp_image)
        end = time.perf_counter()
        times.append((end - start)*1000)
        
    avg_time = sum(times) / runs
    max_time = max(times)
    min_time = min(times)
    
    print(f"MediaPipe Tasks Face Detection Performance (640x360):")
    print(f"  Avg: {avg_time:.2f} ms")
    print(f"  Min: {min_time:.2f} ms")
    print(f"  Max: {max_time:.2f} ms")
    
if __name__ == "__main__":
    test_perf()
