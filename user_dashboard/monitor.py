"""
monitor.py — Sentinel Live Detection Module (Multi-Camera)
Supports multiple concurrent IP camera streams, each with its own
YOLO model instance, detection thread, and state.
Accessed via FastAPI endpoints in api.py.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ── Debug mode (set SENTINEL_DEBUG=1 to enable per-frame logging) ──────────
SENTINEL_DEBUG = os.environ.get("SENTINEL_DEBUG", "0") == "1"

from user_dashboard.spatial_engine import SpatialEngine, extract_bbox_centers
from user_dashboard.tracker import CentroidTracker

import cv2
import numpy as np
import threading
import time
import uuid
from collections import deque

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
CONF_THRESH   = 0.4
IMG_SIZE      = 640
IOU_THRESH    = 0.5
CROWD_SAFE    = 5
CROWD_DANGER  = 10
MAX_SESSIONS  = 8
RECONNECT_ATTEMPTS = 3
RECONNECT_DELAY    = 2.0
INFER_FPS     = 20          # Hardcoded max FPS for smooth performance
JPEG_QUALITY  = 70          # lower = faster encode, smaller frames
INFER_RESIZE  = (640, 360)  # Locked aggressive resize for ultra-low latency inference
DB_LOG_INTERVAL = 2.0       # throttle MongoDB writes to once per N seconds
DIAG_INTERVAL   = 2.0       # print FPS diagnostics every N seconds
GRAB_JPEG_MIN_INTERVAL = 0.040  # throttle grabber JPEG encode (~25 FPS)

MODEL_NAME = "yolov8n.pt"   # switched from yolov8m — 5× faster, pure detection

# ── Face Blurring (Privacy Mode) ───────────────────────────────────────────
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
except Exception as e:
    print(f"Warning: Failed to load face cascade: {e}")
    face_cascade = None


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
    _, buf = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buf.tobytes()


def _blank_frame(msg="No feed — system stopped") -> bytes:
    """Returns a 640×360 dark placeholder JPEG."""
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(img, msg, (60, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2, cv2.LINE_AA)
    return _frame_to_jpeg(img)


# ══════════════════════════════════════════════════════════════════════════
#  CameraSession — per-camera state
# ══════════════════════════════════════════════════════════════════════════

class CameraSession:
    """Encapsulates all state for a single camera stream."""

    def __init__(self, session_id: str, source_url: str, label: str = ""):
        self.id = session_id
        self.source_url = source_url
        self.label = label or source_url

        self.lock = threading.Lock()
        self.running = False
        self.latest_jpeg = None              # raw camera frame (grabber)
        self.latest_annotated_jpeg = None    # annotated frame (inference)
        self.latest_blurred_jpeg = None      # face-blurred frame (inference)
        self.metrics = {
            "people": 0, "risk": "LOW",
            "safety": "SAFE", "audio": "NORMAL",
        }

        self._grabber_thread = None     # frame-grabber thread
        self._inference_thread = None   # YOLO inference thread
        self._cap = None
        self._audio_pa = None
        self._audio_stream = None
        self._model = None

        # ── Atomic frame swap (grabber → inference) ────────────────
        self._grab_lock = threading.Lock()
        self._latest_raw_frame = None   # always the most recent frame

        # Automatic bird's-eye spatial engine (per camera)
        self._spatial = SpatialEngine()

        # Centroid tracker for temporal smoothing
        self._tracker = CentroidTracker()

        # ── AI engine state ────────────────────────────────────────
        self.density_history: deque = deque(maxlen=30)   # (timestamp, people_count)
        self.velocity_smooth: float = 0.0                 # EMA-smoothed velocity
        self.staff_count: int = 0                         # assigned staff

        self.created_at = time.time()
        self.error = None  # last error message

    def to_dict(self) -> dict:
        """Return a JSON-serializable summary."""
        with self.lock:
            return {
                "camera_id": self.id,
                "source_url": self.source_url,
                "label": self.label,
                "running": self.running,
                "error": self.error,
                "created_at": self.created_at,
                "spatial_ready": bool(self._spatial and self._spatial.ready),
                "staff_count": self.staff_count,
                "velocity": round(self.velocity_smooth, 3),
                **dict(self.metrics),
            }


# ══════════════════════════════════════════════════════════════════════════
#  Frame Grabber — fast loop that always keeps the latest frame
# ══════════════════════════════════════════════════════════════════════════

def _grabber_loop(session: CameraSession):
    """Continuously reads frames from the camera as fast as possible.

    Stores only the most recent frame via atomic swap.
    The MJPEG stream endpoint reads session.latest_jpeg which is
    set here (raw frame encoded to JPEG — no inference overlay).
    This guarantees near-real-time streaming regardless of YOLO speed.
    """
    tag = f"[Grabber {session.id[:8]}]"

    # Open video source with buffer disabled
    resolved = int(session.source_url) if str(session.source_url).isdigit() else session.source_url
    cap = None
    for attempt in range(RECONNECT_ATTEMPTS):
        cap = cv2.VideoCapture(resolved)
        if cap.isOpened():
            # Disable internal OpenCV buffering — read always gets latest
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            break
        print(f"{tag} Cannot open source (attempt {attempt + 1}/{RECONNECT_ATTEMPTS}): {session.source_url}")
        cap.release()
        cap = None
        if attempt < RECONNECT_ATTEMPTS - 1:
            time.sleep(RECONNECT_DELAY)

    if cap is None or not cap.isOpened():
        blank = _blank_frame("Cannot open camera source.")
        with session.lock:
            session.latest_jpeg = blank
            session.running = False
            session.error = f"Cannot open source: {session.source_url}"
        print(f"{tag} Failed to open source after {RECONNECT_ATTEMPTS} attempts.")
        return

    with session.lock:
        session._cap = cap
        session.error = None

    print(f"{tag} Frame grabber started for: {session.source_url}")
    disconnect_count = 0
    last_jpeg_time = 0.0  # throttle JPEG encode

    while True:
        with session.lock:
            if not session.running:
                break

        ret, frame = cap.read()
        if not ret:
            disconnect_count += 1
            blank = _blank_frame("Stream disconnected...")
            with session.lock:
                session.latest_jpeg = blank

            if disconnect_count >= RECONNECT_ATTEMPTS:
                print(f"{tag} Stream lost. Attempting reconnect...")
                cap.release()
                cap = cv2.VideoCapture(resolved)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    disconnect_count = 0
                    with session.lock:
                        session._cap = cap
                        session.error = None
                    print(f"{tag} Reconnected.")
                else:
                    with session.lock:
                        session.error = "Stream disconnected — reconnect failed."
                    print(f"{tag} Reconnect failed.")
                    time.sleep(RECONNECT_DELAY)
                    disconnect_count = 0
            else:
                time.sleep(0.5)
            continue

        disconnect_count = 0

        # Atomic swap — inference thread picks this up
        with session._grab_lock:
            session._latest_raw_frame = frame

        # Throttled JPEG encode — only re-encode if enough time has passed
        now = time.time()
        if now - last_jpeg_time >= GRAB_JPEG_MIN_INTERVAL:
            jpeg = _frame_to_jpeg(frame)
            with session.lock:
                session.latest_jpeg = jpeg
            last_jpeg_time = now

        time.sleep(0.005)

    # ── Cleanup ──────────────────────────────────────────────────────────
    cap.release()
    with session.lock:
        session._cap = None
    print(f"{tag} Frame grabber stopped.")


# ══════════════════════════════════════════════════════════════════════════
#  Inference Loop — YOLO + spatial density, capped at INFER_FPS
# ══════════════════════════════════════════════════════════════════════════

def _draw_lightweight_boxes(frame, boxes_np, confs_np):
    """Draw simple rectangles + confidence text — 10× faster than r.plot()."""
    for i, box in enumerate(boxes_np):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        conf = confs_np[i] if i < len(confs_np) else 0.0
        # Green box with 2px border
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Confidence label
        label = f"{conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def _inference_loop(session: CameraSession):
    """Runs YOLO inference on the latest grabbed frame.

    - Reads latest frame via atomic swap (never queues)
    - Capped at INFER_FPS
    - Resizes to INFER_RESIZE before YOLO
    - Lightweight annotation (cv2.rectangle, not r.plot())
    - Throttled DB writes and snapshot publishing
    - FPS diagnostics every DIAG_INTERVAL seconds
    """
    tag = f"[Inference {session.id[:8]}]"

    # Load model (independent instance per session)
    model = None
    if YOLO_AVAILABLE:
        try:
            model = YOLO(MODEL_NAME)

            # ── GPU + half-precision acceleration ─────────────────
            import torch
            if torch.cuda.is_available():
                model.to("cuda")
                model.half()
                print(f"{tag} 🚀 CUDA enabled — running on GPU with FP16")
            else:
                print(f"{tag} ⚠ No CUDA — running on CPU")

            # ── Model integrity logging ───────────────────────────
            print(f"{tag} ══════════════════════════════════════════")
            print(f"{tag} YOLO MODEL DIAGNOSTICS")
            print(f"{tag}   Weight file : {MODEL_NAME}")
            print(f"{tag}   Task        : {model.task}")
            person_cls = [k for k, v in model.names.items() if v == 'person']
            print(f"{tag}   Person class: {person_cls} (filtering on classes=[0])")
            print(f"{tag}   Input size  : {IMG_SIZE}")
            print(f"{tag}   Resize to   : {INFER_RESIZE}")
            print(f"{tag}   Conf thresh : {CONF_THRESH}")
            print(f"{tag}   IOU thresh  : {IOU_THRESH}")
            print(f"{tag}   FPS cap     : {INFER_FPS}")
            print(f"{tag} ══════════════════════════════════════════")
        except Exception as e:
            print(f"{tag} YOLO load error: {e}")

    with session.lock:
        session._model = model

    # Open audio (best-effort, only for webcam)
    audio_stream = None
    pa_instance = None
    if AUDIO_AVAILABLE and str(session.source_url).isdigit():
        try:
            pa_instance = pyaudio.PyAudio()
            audio_stream = pa_instance.open(
                format=pyaudio.paFloat32, channels=1, rate=AUDIO_RATE,
                input=True, frames_per_buffer=AUDIO_CHUNK
            )
            with session.lock:
                session._audio_pa = pa_instance
                session._audio_stream = audio_stream
        except Exception as e:
            print(f"{tag} Audio unavailable: {e}")

    print(f"{tag} Inference loop started (cap: {INFER_FPS} FPS, debug: {SENTINEL_DEBUG})")
    db = shared_db.get_db()
    interval = 1.0 / INFER_FPS

    # ── Throttle / diagnostics state ──────────────────────────────
    last_db_write = 0.0
    last_diag_time = time.time()
    diag_frame_count = 0
    diag_infer_total = 0.0

    while True:
        t0 = time.time()

        with session.lock:
            if not session.running:
                break

        # Grab the latest frame (atomic, skip old frames)
        with session._grab_lock:
            frame = session._latest_raw_frame

        if frame is None:
            time.sleep(0.05)  # no frame yet, wait
            continue

        orig_h, orig_w = frame.shape[:2]

        # ── Frame resize for inference ────────────────────────────
        # Resize to INFER_RESIZE to reduce compute. Keep orig dims
        # for coordinate mapping.
        target_w, target_h = INFER_RESIZE
        if orig_w > target_w or orig_h > target_h:
            infer_frame = cv2.resize(frame, (target_w, target_h),
                                     interpolation=cv2.INTER_LINEAR)
        else:
            infer_frame = frame
        h, w = infer_frame.shape[:2]

        if SENTINEL_DEBUG:
            print(f"{tag} Frame: {orig_w}×{orig_h} → {w}×{h}")

        # ── YOLO inference + tracking ─────────────────────────────
        annotated = infer_frame  # no copy — we draw directly
        tracker = session._tracker
        boxes_np = np.empty((0, 4))
        confs_np = np.array([])

        t_infer_start = time.time()

        if model is not None:
            try:
                results = model(infer_frame, verbose=False, classes=[0],
                                conf=CONF_THRESH, imgsz=IMG_SIZE, iou=IOU_THRESH)
                for r in results:
                    if r.boxes and len(r.boxes) > 0:
                        boxes_np = r.boxes.xyxy.cpu().numpy()
                        confs_np = r.boxes.conf.cpu().numpy()

                        # Scale boxes back to original frame coordinates
                        if orig_w != w or orig_h != h:
                            scale_x = orig_w / w
                            scale_y = orig_h / h
                            boxes_np = boxes_np.copy()
                            boxes_np[:, [0, 2]] *= scale_x
                            boxes_np[:, [1, 3]] *= scale_y

                        if SENTINEL_DEBUG:
                            print(f"{tag}   Raw detections: {len(boxes_np)}")

                        tracker.update(boxes_np, confs_np)
                    else:
                        tracker.update(np.empty((0, 4)))
                        if SENTINEL_DEBUG:
                            print(f"{tag}   Raw detections: 0")
            except Exception as e:
                print(f"{tag} Inference error: {e}")
                tracker.update(np.empty((0, 4)))
        else:
            tracker.update(np.empty((0, 4)))

        t_infer_end = time.time()
        infer_ms = (t_infer_end - t_infer_start) * 1000

        # ── Lightweight annotation (replaces r.plot()) ────────────
        # Draw on original-res frame for MJPEG output
        annotated = frame.copy() if len(boxes_np) > 0 else frame
        if len(boxes_np) > 0:
            annotated = _draw_lightweight_boxes(annotated, boxes_np, confs_np)

        # ── Tracked metrics ───────────────────────────────────────
        p_count = tracker.active_count
        raw_centers = extract_bbox_centers(boxes_np)

        if SENTINEL_DEBUG:
            print(f"{tag}   Tracked: active={tracker.active_count} total={tracker.total_count}")

        # ── Public face-blurred frame ─────────────────────────────
        blurred_frame = annotated.copy()
        if face_cascade is not None and len(boxes_np) > 0:
            try:
                gray = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2GRAY)
                # Quick face detection on the current frame
                faces = face_cascade.detectMultiScale(gray, 1.2, 5)
                for (x, y, w_face, h_face) in faces:
                    # Extract ROI, blur it heavily, put it back
                    face_roi = blurred_frame[y:y+h_face, x:x+w_face]
                    face_roi = cv2.GaussianBlur(face_roi, (51, 51), 30)
                    blurred_frame[y:y+h_face, x:x+w_face] = face_roi
            except Exception as e:
                # If blurring fails for any reason, don't crash loop
                print(f"{tag} Face blur error: {e}")

        # ── Spatial projection pipeline ───────────────────────────
        spatial = session._spatial
        spatial.offer_frame(frame)
        spatial.update(raw_centers)

        # ── Audio analysis ────────────────────────────────────────
        aud_status = _analyze_audio(audio_stream)

        # ── Risk fusion ───────────────────────────────────────────
        risk, safety = _get_risk(aud_status, p_count)

        # ── Store annotated frame + metrics ───────────────────────
        annotated_jpeg = _frame_to_jpeg(annotated)
        blurred_jpeg = _frame_to_jpeg(blurred_frame)
        now = time.time()
        with session.lock:
            session.latest_annotated_jpeg = annotated_jpeg
            session.latest_blurred_jpeg = blurred_jpeg
            session.metrics = {
                "people": p_count,
                "risk": risk,
                "safety": safety,
                "audio": aud_status,
            }

            # ── AI: record density history + compute velocity ─────
            session.density_history.append((now, p_count))
            if len(session.density_history) >= 2:
                ts_prev, d_prev = session.density_history[-2]
                dt = now - ts_prev
                if dt > 0.01:
                    v_new = (p_count - d_prev) / dt
                    alpha = 0.4
                    session.velocity_smooth = (
                        alpha * v_new + (1.0 - alpha) * session.velocity_smooth
                    )

        # ── Throttled MongoDB write (every DB_LOG_INTERVAL sec) ───
        if now - last_db_write >= DB_LOG_INTERVAL:
            try:
                db.log_event(p_count, aud_status, risk)
            except Exception:
                pass
            last_db_write = now

        # ── FPS Diagnostics (every DIAG_INTERVAL sec) ─────────────
        diag_frame_count += 1
        diag_infer_total += infer_ms
        elapsed = time.time() - t0
        if now - last_diag_time >= DIAG_INTERVAL:
            avg_infer = diag_infer_total / max(diag_frame_count, 1)
            fps = diag_frame_count / max(now - last_diag_time, 0.001)
            pipeline_ms = elapsed * 1000
            print(f"{tag} 📊 Inference: {avg_infer:.1f}ms | "
                  f"Pipeline: {pipeline_ms:.1f}ms | "
                  f"FPS: {fps:.1f} | "
                  f"People: {p_count}")
            diag_frame_count = 0
            diag_infer_total = 0.0
            last_diag_time = now

        # ── FPS cap ───────────────────────────────────────────────
        sleep_time = interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    # ── Cleanup ──────────────────────────────────────────────────────────
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
    with session.lock:
        session._audio_stream = None
        session._audio_pa = None
        session._model = None

    print(f"{tag} Inference loop stopped.")


# ══════════════════════════════════════════════════════════════════════════
#  MultiCameraManager — thread-safe session registry
# ══════════════════════════════════════════════════════════════════════════

class MultiCameraManager:
    """Manages multiple concurrent camera sessions."""

    def __init__(self):
        self._sessions: dict[str, CameraSession] = {}
        # Lightweight basic room mapping
        self._rooms = {
            "Room 1": [],
            "Room 2": []
        }
        self._lock = threading.Lock()

    def add_camera(self, source_url: str, label: str = "", room: str = "Room 1") -> tuple[str, str]:
        """Register a new camera. Returns (camera_id, error_msg)."""
        with self._lock:
            if len(self._sessions) >= MAX_SESSIONS:
                return "", f"Maximum of {MAX_SESSIONS} concurrent cameras reached."
            camera_id = uuid.uuid4().hex[:12]
            session = CameraSession(camera_id, source_url, label)
            self._sessions[camera_id] = session
            if room in self._rooms:
                self._rooms[room].append(camera_id)
        return camera_id, ""

    def remove_camera(self, camera_id: str) -> tuple[bool, str]:
        """Stop and remove a camera session. Returns (success, error_msg)."""
        with self._lock:
            session = self._sessions.get(camera_id)
            if session is None:
                return False, "Camera not found."

        # Stop first if running
        self.stop_camera(camera_id)

        # Wait for both threads to finish
        for t in (session._grabber_thread, session._inference_thread):
            if t and t.is_alive():
                t.join(timeout=3.0)

        with self._lock:
            self._sessions.pop(camera_id, None)
            for r_cams in self._rooms.values():
                if camera_id in r_cams:
                    r_cams.remove(camera_id)
        return True, ""

    def start_camera(self, camera_id: str) -> tuple[bool, str]:
        """Start detection on a specific camera. Returns (success, error_msg)."""
        with self._lock:
            session = self._sessions.get(camera_id)
            if session is None:
                return False, "Camera not found."

        with session.lock:
            if session.running:
                return False, "Already running."
            session.running = True
            session.error = None
            session.metrics = {
                "people": 0, "risk": "LOW",
                "safety": "SAFE", "audio": "NORMAL",
            }
            session.latest_jpeg = _blank_frame("Starting up...")
            session._spatial = SpatialEngine()
            session._tracker.reset()

        # Launch grabber thread (fast frame reader)
        gt = threading.Thread(
            target=_grabber_loop,
            args=(session,),
            daemon=True,
            name=f"sentinel-grab-{camera_id}"
        )
        gt.start()

        # Launch inference thread (YOLO at INFER_FPS)
        it = threading.Thread(
            target=_inference_loop,
            args=(session,),
            daemon=True,
            name=f"sentinel-infer-{camera_id}"
        )
        it.start()

        with session.lock:
            session._grabber_thread = gt
            session._inference_thread = it

        return True, ""

    def stop_camera(self, camera_id: str) -> tuple[bool, str]:
        """Stop detection on a specific camera. Returns (success, error_msg)."""
        with self._lock:
            session = self._sessions.get(camera_id)
            if session is None:
                return False, "Camera not found."

        with session.lock:
            if not session.running:
                return False, "Not running."
            session.running = False
        return True, ""

    def get_session(self, camera_id: str) -> CameraSession | None:
        """Get a session by ID."""
        with self._lock:
            return self._sessions.get(camera_id)

    def list_sessions(self) -> list[dict]:
        """Return summary dicts for all sessions."""
        with self._lock:
            sessions = list(self._sessions.values())
        return [s.to_dict() for s in sessions]

    def get_frame(self, camera_id: str, stream_type: str = "admin") -> bytes:
        """Get latest JPEG for a camera based on stream type.
        stream_type 'admin': returns annotated > raw
        stream_type 'public': returns blurred > annotated > raw
        """
        with self._lock:
            session = self._sessions.get(camera_id)
        if session is None:
            return _blank_frame("Camera not found.")
        
        with session.lock:
            if stream_type == "public":
                return session.latest_blurred_jpeg or session.latest_annotated_jpeg or session.latest_jpeg or _blank_frame()
            else:
                return session.latest_annotated_jpeg or session.latest_jpeg or _blank_frame()

    def get_spatial_data(self, camera_id: str) -> dict:
        """Get spatial density snapshot for a camera."""
        empty = {
            "grid": [], "points": [],
            "width": 0, "height": 0,
            "ts": 0,
            "ready": False
        }
        with self._lock:
            session = self._sessions.get(camera_id)
        if session is None:
            return empty
        sf = session._spatial
        if sf is None:
            return empty
        snap = sf.get_snapshot()
        return {
            "grid": snap.grid.tolist(),
            "points": snap.points.tolist(),
            "width": snap.width,
            "height": snap.height,
            "ts": snap.timestamp,
            "ready": snap.ready,
        }

    def get_room_spatial_data(self, room_name: str) -> dict:
        """Fuses spatial density grids for all cameras in the room."""
        empty = {
            "grid": [], "points": [],
            "width": 0, "height": 0,
            "ts": 0,
            "ready": False
        }
        with self._lock:
            cam_ids = self._rooms.get(room_name, [])
            sessions = [self._sessions.get(cid) for cid in cam_ids if self._sessions.get(cid) is not None]

        if not sessions:
            return empty

        # Grab snapshots independently
        snaps = []
        for session in sessions:
            sf = session._spatial
            if sf is not None:
                snap = sf.get_snapshot()
                if snap.ready:
                    snaps.append(snap)

        if not snaps:
            return empty

        # Fuse
        # Use dimensions of the first valid config
        base_w = snaps[0].width
        base_h = snaps[0].height
        base_ts = snaps[0].timestamp

        fused_grid = np.zeros_like(snaps[0].grid)
        fused_points = []

        for s in snaps:
            if s.grid.shape == fused_grid.shape:
                fused_grid += s.grid
            if len(s.points) > 0:
                fused_points.extend(s.points.tolist())
            base_ts = max(base_ts, s.timestamp)

        return {
            "grid": fused_grid.tolist(),
            "points": fused_points,
            "width": base_w,
            "height": base_h,
            "ts": base_ts,
            "ready": True
        }

    def calibrate_spatial(self, camera_id: str, camera_pts: list[list[float]], blueprint_pts: list[list[float]], bp_w: int, bp_h: int) -> tuple[bool, str]:
        """Set the 4-point homography calibration for a camera's spatial engine."""
        with self._lock:
            session = self._sessions.get(camera_id)
        if session is None:
            return False, "Camera not found."
        sf = session._spatial
        if sf is None:
            return False, "Spatial engine not initialized."
        ok = sf.set_calibration(camera_pts, blueprint_pts, bp_w, bp_h)
        if not ok:
            return False, "Failed to compute calibration."
        return True, ""

    def cleanup_dead(self) -> int:
        """Reap sessions whose threads have crashed. Returns count removed."""
        to_remove = []
        with self._lock:
            for cid, session in self._sessions.items():
                with session.lock:
                    gt = session._grabber_thread
                    it = session._inference_thread
                    dead = (
                        (gt and not gt.is_alive()) or
                        (it and not it.is_alive())
                    ) and not session.running
                    if dead:
                        to_remove.append(cid)
        for cid in to_remove:
            with self._lock:
                self._sessions.pop(cid, None)
        return len(to_remove)

    def stop_all(self) -> int:
        """Stop all running cameras. Returns count stopped."""
        count = 0
        with self._lock:
            sessions = list(self._sessions.values())
        for s in sessions:
            with s.lock:
                if s.running:
                    s.running = False
                    count += 1
        return count

    def start_all(self) -> int:
        """Start all stopped cameras. Returns count started."""
        count = 0
        with self._lock:
            session_ids = list(self._sessions.keys())
        for cid in session_ids:
            ok, _ = self.start_camera(cid)
            if ok:
                count += 1
        return count


# ── Global manager instance ────────────────────────────────────────────────
_manager = MultiCameraManager()


# ══════════════════════════════════════════════════════════════════════════
#  Public API — Multi-camera
# ══════════════════════════════════════════════════════════════════════════

def get_manager() -> MultiCameraManager:
    """Return the global MultiCameraManager."""
    return _manager


# ══════════════════════════════════════════════════════════════════════════
#  Legacy Public API — backward-compatible single-camera interface
#  Delegates to manager using a "default" session
# ══════════════════════════════════════════════════════════════════════════

_default_camera_id: str | None = None
_default_lock = threading.Lock()


def _find_running_session_id() -> str | None:
    """Find any running camera session in the manager."""
    sessions = _manager.list_sessions()
    for s in sessions:
        if s.get("running"):
            return s["camera_id"]
    # If none running, return first session if exists
    if sessions:
        return sessions[0]["camera_id"]
    return None


def start(source="0", model_name: str = "yolov8m.pt") -> bool:
    """Start the detection loop (legacy). Returns False if already running."""
    global _default_camera_id
    with _default_lock:
        # Validate current default — clear if stale
        if _default_camera_id is not None:
            session = _manager.get_session(_default_camera_id)
            if session is None:
                _default_camera_id = None  # session was removed

        # Try to adopt an existing running session
        if _default_camera_id is None:
            existing = _find_running_session_id()
            if existing is not None:
                _default_camera_id = existing
                session = _manager.get_session(existing)
                if session is not None:
                    with session.lock:
                        if session.running:
                            return True  # Already running, adopt it

        # Check if current default is already running
        if _default_camera_id is not None:
            session = _manager.get_session(_default_camera_id)
            if session is not None:
                with session.lock:
                    if session.running:
                        return True  # Already running

        # Create or reuse default session
        if _default_camera_id is None:
            cid, err = _manager.add_camera(source, label="Default")
            if err:
                return False
            _default_camera_id = cid
        else:
            # Update source if changed
            session = _manager.get_session(_default_camera_id)
            if session:
                session.source_url = source

    ok, _ = _manager.start_camera(_default_camera_id)
    return ok


def stop() -> bool:
    """Stop the detection loop (legacy). Returns False if not running."""
    with _default_lock:
        if _default_camera_id is None:
            return False
    ok, _ = _manager.stop_camera(_default_camera_id)
    return ok


def is_running() -> bool:
    global _default_camera_id
    with _default_lock:
        if _default_camera_id is None:
            # Check if any session is running via multi-camera API
            existing = _find_running_session_id()
            if existing is not None:
                _default_camera_id = existing
            else:
                return False
    session = _manager.get_session(_default_camera_id)
    if session is None:
        return False
    with session.lock:
        return session.running


def get_metrics() -> dict:
    with _default_lock:
        cid = _default_camera_id
    if cid is None:
        return {"people": 0, "risk": "LOW", "safety": "SAFE", "audio": "NORMAL"}
    session = _manager.get_session(cid)
    if session is None:
        return {"people": 0, "risk": "LOW", "safety": "SAFE", "audio": "NORMAL"}
    with session.lock:
        return dict(session.metrics)


def get_latest_jpeg() -> bytes:
    """Returns latest annotated JPEG bytes, or a blank placeholder."""
    global _default_camera_id
    with _default_lock:
        cid = _default_camera_id
        if cid is None:
            existing = _find_running_session_id()
            if existing is not None:
                _default_camera_id = existing
                cid = existing
    if cid is None:
        return _blank_frame()
    return _manager.get_frame(cid)  # get_frame now prefers annotated


def get_spatial_data() -> dict:
    """Returns latest spatial density data for the default camera."""
    global _default_camera_id
    with _default_lock:
        cid = _default_camera_id
        if cid is None:
            existing = _find_running_session_id()
            if existing is not None:
                _default_camera_id = existing
                cid = existing
    empty = {
        "grid": [], "points": [],
        "width": 0, "height": 0,
        "ts": 0,
    }
    if cid is None:
        return empty
    return _manager.get_spatial_data(cid)


def get_spatial_background() -> bytes | None:
    """Return the top-view background JPEG for the default camera."""
    global _default_camera_id
    with _default_lock:
        cid = _default_camera_id
        if cid is None:
            existing = _find_running_session_id()
            if existing is not None:
                _default_camera_id = existing
                cid = existing
    if cid is None:
        return None
    return _manager.get_spatial_background(cid)
