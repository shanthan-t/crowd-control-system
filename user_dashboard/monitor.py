"""
monitor.py — Sentinel Live Detection Module
Extracted from control_room/app.py. Zero Streamlit dependencies.
Runs YOLO detection + audio analysis in a background thread.
Accessed via FastAPI endpoints in api.py.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from user_dashboard import calibration as calib_mod

import cv2
import numpy as np
import threading
import time
import io

try:
    import pyaudio
    import librosa
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

from shared import db as shared_db

# ── Configuration ──────────────────────────────────────────────────────────
AUDIO_RATE    = 22050
AUDIO_CHUNK   = 1024
CONF_THRESH   = 0.25
IMG_SIZE      = 640
IOU_THRESH    = 0.45
CROWD_SAFE    = 5
CROWD_DANGER  = 10

MODEL_MAP = {
    "yolov8n-pose.pt": "yolov8n-pose.pt",
}

# ── Global singleton state ─────────────────────────────────────────────────
class _MonitorState:
    def __init__(self):
        self.lock        = threading.Lock()
        self.running     = False
        self.latest_jpeg = None   # raw JPEG bytes of latest annotated frame
        self.metrics     = {
            "people":   0,
            "risk":     "LOW",
            "safety":   "SAFE",
            "audio":    "NORMAL",
            "skeleton": False,
        }
        self._thread      = None
        self._cap         = None
        self._audio_pa    = None
        self._audio_stream= None
        self._model       = None

        # ── Heatmap / calibration state ────────────────────────────
        self._calib           = None   # CalibSnapshot (frozen, atomic swap)
        self.heatmap_positions = []    # transformed blueprint coords
        self.heatmap_grid      = []    # density grid
        self.heatmap_ts        = 0     # timestamp of last update

_state = _MonitorState()


# ── Helpers ────────────────────────────────────────────────────────────────

def _get_risk(audio_status: str, count: int):
    if count >= CROWD_DANGER:
        return "HIGH", "DANGEROUS"
    if audio_status == "PANIC" or count >= CROWD_SAFE:
        return "MEDIUM", "CAUTION"
    return "LOW", "SAFE"


def _analyze_audio(stream) -> str:
    """Returns 'PANIC' or 'NORMAL'."""
    if not AUDIO_AVAILABLE or stream is None:
        return "NORMAL"
    try:
        if stream.get_read_available() >= AUDIO_CHUNK:
            data = stream.read(AUDIO_CHUNK, exception_on_overflow=False)
            y = np.frombuffer(data, dtype=np.float32)
            rms  = float(np.mean(librosa.feature.rms(y=y)))
            cent = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=AUDIO_RATE)))
            if rms > 0.05 and cent > 2000:
                return "PANIC"
    except Exception:
        pass
    return "NORMAL"


def _frame_to_jpeg(frame_bgr) -> bytes:
    _, buf = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return buf.tobytes()


def _blank_frame(msg="No feed — system stopped") -> bytes:
    """Returns a 640×360 dark placeholder JPEG."""
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(img, msg, (60, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2, cv2.LINE_AA)
    return _frame_to_jpeg(img)


# ── Detection loop (runs in daemon thread) ─────────────────────────────────

def _detection_loop(source, model_name: str):
    global _state

    # 1. Load model
    model = None
    if YOLO_AVAILABLE:
        try:
            model = YOLO(model_name)
        except Exception as e:
            print(f"[Monitor] YOLO load error: {e}")

    # 2. Open video source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[Monitor] Cannot open source: {source}")
        # emit blank frames so stream stays responsive
        blank = _blank_frame("Cannot open camera source.")
        with _state.lock:
            _state.latest_jpeg = blank
            _state.running = False
        return

    with _state.lock:
        _state._cap = cap

    # 3. Open audio (best-effort)
    audio_stream = None
    pa_instance  = None
    if AUDIO_AVAILABLE:
        try:
            pa_instance  = pyaudio.PyAudio()
            audio_stream = pa_instance.open(
                format=pyaudio.paFloat32, channels=1, rate=AUDIO_RATE,
                input=True, frames_per_buffer=AUDIO_CHUNK
            )
            with _state.lock:
                _state._audio_pa     = pa_instance
                _state._audio_stream = audio_stream
        except Exception as e:
            print(f"[Monitor] Audio unavailable: {e}")

    print("[Monitor] Detection loop started.")
    db = shared_db.get_db()

    while True:
        with _state.lock:
            if not _state.running:
                break

        ret, frame = cap.read()
        if not ret:
            # Stream dropped — emit blank and try to reconnect
            blank = _blank_frame("Stream disconnected...")
            with _state.lock:
                _state.latest_jpeg = blank
            time.sleep(0.5)
            continue

        # 4. YOLO inference
        annotated = frame.copy()
        p_count   = 0
        has_skel  = False

        raw_centers = []  # bbox bottom-centers for heatmap

        if model is not None:
            try:
                results = model(frame, verbose=False, classes=[0],
                                conf=CONF_THRESH, imgsz=IMG_SIZE, iou=IOU_THRESH)
                for r in results:
                    annotated = r.plot()
                    if r.boxes:
                        p_count += len(r.boxes)
                        # Extract bottom-center (feet) for heatmap
                        raw_centers = calib_mod.extract_bbox_centers(
                            r.boxes.xyxy.cpu().numpy()
                        )
                    if r.keypoints is not None and len(r.keypoints) > 0:
                        has_skel = True
            except Exception as e:
                print(f"[Monitor] Inference error: {e}")

        # 4b. Heatmap pipeline (if calibrated)
        calib = _state._calib  # atomic read — immutable snapshot
        if calib is not None and raw_centers:
            filtered = calib_mod.filter_inside_quad(raw_centers, calib.camera_quad)
            projected = calib_mod.transform_points(filtered, calib.H)
            grid = calib_mod.build_density_grid(
                projected, calib.bp_width, calib.bp_height
            )
        else:
            projected = []
            grid = []

        # 5. Audio analysis
        aud_status = _analyze_audio(audio_stream)

        # 6. Risk fusion
        risk, safety = _get_risk(aud_status, p_count)

        # 7. Store
        jpeg = _frame_to_jpeg(annotated)
        with _state.lock:
            _state.latest_jpeg = jpeg
            _state.metrics = {
                "people":   p_count,
                "risk":     risk,
                "safety":   safety,
                "audio":    aud_status,
                "skeleton": has_skel,
            }
            # Heatmap data
            if calib is not None:
                _state.heatmap_positions = projected
                _state.heatmap_grid = grid
                _state.heatmap_ts = time.time()

        # 8. Persist to MongoDB (throttled — every 2s approx)
        try:
            db.log_event(p_count, aud_status, risk)
        except Exception:
            pass

        time.sleep(0.04)  # ~25 fps ceiling

    # ── Cleanup ──────────────────────────────────────────────────────────
    cap.release()
    if audio_stream:
        try:
            audio_stream.stop_stream()
            audio_stream.close()
        except Exception:
            pass
    if pa_instance:
        try:
            pa_instance.terminate()
        except Exception:
            pass
    with _state.lock:
        _state._cap          = None
        _state._audio_stream = None
        _state._audio_pa     = None
        _state.latest_jpeg   = None

    print("[Monitor] Detection loop stopped.")


# ── Public API ─────────────────────────────────────────────────────────────

def start(source="0", model_name: str = "yolov8n-pose.pt") -> bool:
    """Start the detection loop. Returns False if already running."""
    with _state.lock:
        if _state.running:
            return False
        _state.running = True
        _state.metrics = {
            "people": 0, "risk": "LOW",
            "safety": "SAFE", "audio": "NORMAL", "skeleton": False,
        }
        _state.latest_jpeg = _blank_frame("Starting up...")

    # Load calibration if available (atomic snapshot)
    snap, _ = calib_mod.load_calibration()
    with _state.lock:
        _state._calib = snap  # None if not calibrated

    # Resolve source
    resolved = int(source) if str(source).isdigit() else source

    t = threading.Thread(
        target=_detection_loop,
        args=(resolved, model_name),
        daemon=True,
        name="sentinel-monitor"
    )
    t.start()
    with _state.lock:
        _state._thread = t

    return True


def stop() -> bool:
    """Stop the detection loop. Returns False if not running."""
    with _state.lock:
        if not _state.running:
            return False
        _state.running = False
    return True


def is_running() -> bool:
    with _state.lock:
        return _state.running


def get_metrics() -> dict:
    with _state.lock:
        return dict(_state.metrics)


def get_latest_jpeg() -> bytes:
    """Returns latest annotated JPEG bytes, or a blank placeholder."""
    with _state.lock:
        return _state.latest_jpeg or _blank_frame()


def get_heatmap_data() -> dict:
    """Returns latest heatmap positions, grid, and timestamp."""
    with _state.lock:
        return {
            "positions": list(_state.heatmap_positions),
            "grid": list(_state.heatmap_grid),
            "ts": _state.heatmap_ts,
            "people": len(_state.heatmap_positions),
        }


def set_calib(snap) -> None:
    """Atomically swap the calibration snapshot (called from API thread)."""
    with _state.lock:
        _state._calib = snap


def get_calib():
    """Return current CalibSnapshot or None."""
    with _state.lock:
        return _state._calib
