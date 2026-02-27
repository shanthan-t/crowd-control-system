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
from user_dashboard.face_tracker import FaceMeshTracker

import cv2
import numpy as np
import threading
import time
import uuid
import queue
from collections import deque

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

from shared import db as shared_db

# ── Configuration ──────────────────────────────────────────────────────────
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
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    
    _model_path = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
    _base_options = python.BaseOptions(model_asset_path=_model_path)
    _options = vision.FaceLandmarkerOptions(
        base_options=_base_options,
        num_faces=10,
        min_face_detection_confidence=0.4,
        min_face_presence_confidence=0.4,
        min_tracking_confidence=0.4
    )
    face_mesh = vision.FaceLandmarker.create_from_options(_options)
    FACE_AVAILABLE = True
except Exception as e:
    print(f"Warning: Failed to load MediaPipe Face Mesh: {e}")
    FACE_AVAILABLE = False


# ── Helpers ────────────────────────────────────────────────────────────────

def _get_risk(count: int):
    if count >= CROWD_DANGER:
        return "HIGH", "DANGEROUS"
    if count >= CROWD_SAFE:
        return "MEDIUM", "CAUTION"
    return "LOW", "SAFE"


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
            "safety": "SAFE",
        }

        self._grabber_thread = None     # frame-grabber thread
        self._inference_thread = None   # YOLO inference thread
        self._cap = None
        self._model = None

        # ── 3-Stage Pipeline State ────────────────
        self._grab_lock = threading.Lock()
        self._latest_raw_frame = None

        self._infer_lock = threading.Lock()
        # Stores: (boxes_np, confs_np, face_lms_smoothed, raw_centers, p_count)
        self._latest_infer_res = (np.empty((0, 4)), np.array([]), [], [], 0)
        
        self.stream_event = threading.Condition()

        # Automatic bird's-eye spatial engine (per camera)
        self._spatial = SpatialEngine()

        # Centroid tracker for temporal smoothing
        self._tracker = CentroidTracker()
        self._face_tracker = FaceMeshTracker(max_disappeared=10, match_radius=100.0)

        # ── AI engine state ────────────────────────────────────────
        self.density_history: deque = deque(maxlen=30)   # (timestamp, people_count)
        self.velocity_smooth: float = 0.0                 # EMA-smoothed velocity
        self.staff_count: int = 0                         # assigned staff
        
        # ── FPS Metrics ──────────────────────────────────────────
        self.fps_metrics = {"grab": 0.0, "infer": 0.0, "stream": 0.0}
        
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
                "fps": self.fps_metrics,
                **dict(self.metrics),
            }


# ══════════════════════════════════════════════════════════════════════════
#  Frame Grabber — fast loop that always keeps the latest frame
# ══════════════════════════════════════════════════════════════════════════

def _grabber_loop(session: CameraSession):
    """STAGE 1: Capture Thread
    Continuously reads frames as fast as possible. No inference.
    Stores only the most recent frame via grab_lock.
    """
    tag = f"[Grabber {session.id[:8]}]"

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
    resolved = int(session.source_url) if str(session.source_url).isdigit() else session.source_url
    cap = None
    
    def _connect():
        nonlocal cap
        for attempt in range(RECONNECT_ATTEMPTS):
            cap = cv2.VideoCapture(resolved, cv2.CAP_FFMPEG) if isinstance(resolved, str) else cv2.VideoCapture(resolved)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return True
            print(f"{tag} Cannot open source (attempt {attempt + 1}/{RECONNECT_ATTEMPTS}): {session.source_url}")
            cap.release()
            cap = None
            if attempt < RECONNECT_ATTEMPTS - 1:
                time.sleep(RECONNECT_DELAY)
        return False

    if not _connect():
        with session.lock:
            session.latest_jpeg = _blank_frame("Cannot open camera source.")
            session.running = False
            session.error = f"Cannot open source: {session.source_url}"
        print(f"{tag} Failed to open source.")
        return

    with session.lock:
        session._cap = cap
        session.error = None

    print(f"{tag} Frame grabber started for: {session.source_url}")
    
    # FPS Tracker
    frame_count = 0
    t0 = time.time()

    while True:
        with session.lock:
            if not session.running:
                break

        ret, frame = cap.read()
        if not ret:
            with session.lock:
                session.latest_jpeg = _blank_frame("Stream disconnected... Reconnecting...")
            print(f"{tag} Stream lost. Attempting reconnect...")
            cap.release()
            time.sleep(0.5)
            if _connect():
                with session.lock:
                    session._cap = cap
                    session.error = None
                print(f"{tag} Reconnected.")
            else:
                with session.lock:
                    session.error = "Stream disconnected — reconnect failed."
                print(f"{tag} Reconnect failed.")
                time.sleep(RECONNECT_DELAY)
            continue
            
        with session._grab_lock:
            session._latest_raw_frame = frame
            
        frame_count += 1
        now = time.time()
        if now - t0 >= 5.0:
            session.fps_metrics["grab"] = round(frame_count / (now - t0), 1)
            frame_count = 0
            t0 = now

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
    """STAGE 2: Inference Worker
    Reads _latest_raw_frame as snapshot.
    Runs YOLO async without blocking the stream or the capture thread.
    Saves output to _latest_infer_res via _infer_lock.
    """
    tag = f"[Inference {session.id[:8]}]"
    
    model = None
    if YOLO_AVAILABLE:
        try:
            model = YOLO(MODEL_NAME)
            import torch
            if torch.cuda.is_available():
                model.to("cuda")
                model.half()
        except Exception as e:
            print(f"{tag} YOLO load error: {e}")

    with session.lock:
        session._model = model

    print(f"{tag} Inference worker started (Target: {INFER_FPS} FPS)")
    
    tracker = session._tracker
    db = shared_db.get_db()
    
    last_db_write = 0.0
    def _db_logger():
        nonlocal last_db_write
        while getattr(session, "running", False):
            now = time.time()
            if now - last_db_write >= DB_LOG_INTERVAL:
                with session.lock:
                    p = session.metrics.get("people", 0)
                    r = session.metrics.get("risk", "LOW")
                try:
                    db.log_event(p, "NORMAL", r) # "NORMAL" for audio status as it's removed
                except Exception:
                    pass
                last_db_write = now
            time.sleep(0.5)
            
    threading.Thread(target=_db_logger, daemon=True).start()

    frame_count = 0
    t0 = time.time()
    
    # Store reference to prevent running inference on exact same frame twice
    last_processed_frame = None

    while True:
        with session.lock:
            if not session.running:
                break
                
        # 1. Grab snapshot of latest frame
        with session._grab_lock:
            frame = session._latest_raw_frame
            
        if frame is None or frame is last_processed_frame:
            time.sleep(0.01)
            continue
            
        last_processed_frame = frame
        orig_h, orig_w = frame.shape[:2]

        target_w, target_h = INFER_RESIZE
        if orig_w > target_w or orig_h > target_h:
            infer_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        else:
            infer_frame = frame
        h, w = infer_frame.shape[:2]

        boxes_np = np.empty((0, 4))
        confs_np = np.array([])

        if model is not None:
            try:
                results = model(infer_frame, verbose=False, classes=[0],
                                conf=CONF_THRESH, imgsz=IMG_SIZE, iou=IOU_THRESH)
                for r in results:
                    if r.boxes and len(r.boxes) > 0:
                        boxes_np = r.boxes.xyxy.cpu().numpy()
                        confs_np = r.boxes.conf.cpu().numpy()

                        if orig_w != w or orig_h != h:
                            scale_x = orig_w / w
                            scale_y = orig_h / h
                            boxes_np = boxes_np.copy()
                            boxes_np[:, [0, 2]] *= scale_x
                            boxes_np[:, [1, 3]] *= scale_y

                        tracker.update(boxes_np, confs_np)
                    else:
                        tracker.update(np.empty((0, 4)))
            except Exception:
                tracker.update(np.empty((0, 4)))
        else:
            tracker.update(np.empty((0, 4)))

        # Run MediaPipe face mesh detection on infer_frame
        face_lms_smoothed = []
        if FACE_AVAILABLE:
            try:
                rgb_frame = cv2.cvtColor(infer_frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                res = face_mesh.detect(mp_image)
                
                face_landmarks_list = []
                if res.face_landmarks:
                    for face_lm in res.face_landmarks:
                        lms = []
                        for lm in face_lm:
                            lms.append([lm.x * orig_w, lm.y * orig_h])
                        face_landmarks_list.append(np.array(lms, dtype=np.float64))
                        
                session._face_tracker.update(face_landmarks_list)
            except Exception as e:
                session._face_tracker.update([])
        else:
            session._face_tracker.update([])
            
        face_lms_smoothed = session._face_tracker.get_smoothed_landmarks()

        # 2. Process results
        p_count = tracker.active_count
        raw_centers = extract_bbox_centers(boxes_np)
        risk, safety = _get_risk(p_count)
        
        # 3. Save inference state to shared lock
        with session._infer_lock:
            session._latest_infer_res = (boxes_np, confs_np, face_lms_smoothed, raw_centers, p_count)
            
        # 4. Save metrics state
        now = time.time()
        with session.lock:
            session.metrics = {
                "people": p_count,
                "risk": risk,
                "safety": safety,
            }
            session.density_history.append((now, p_count))
            if len(session.density_history) >= 2:
                ts_prev, d_prev = session.density_history[-2]
                dt = now - ts_prev
                if dt > 0.01:
                    v_new = (p_count - d_prev) / dt
                    alpha = 0.4
                    session.velocity_smooth = alpha * v_new + (1.0 - alpha) * session.velocity_smooth
                    
        frame_count += 1
        if now - t0 >= 5.0:
            session.fps_metrics["infer"] = round(frame_count / (now - t0), 1)
            frame_count = 0
            t0 = now
            import psutil
            cpu = psutil.cpu_percent()
            print(f"📊 [Monitor {session.id[:4]}] FPS -> Cap: {session.fps_metrics['grab']} | Inf: {session.fps_metrics['infer']} | Stream: {session.fps_metrics['stream']} | CPU: {cpu}%")
            
        # Cap inference to INFER_FPS (leave CPU room for Streamer)
        time.sleep(1.0 / INFER_FPS)

    print(f"{tag} Inference worker stopped.")


def _stream_loop(session: CameraSession):
    """STAGE 3: Stream Output Worker
    Reads _latest_raw_frame + _latest_infer_res.
    Always generates output at 30fps. Never blocks on YOLO.
    Triggers stream_event for FastAPI MJPEG yield.
    """
    tag = f"[Streamer {session.id[:8]}]"
    print(f"{tag} Stream output worker started")
    
    frame_count = 0
    t0 = time.time()

    while True:
        with session.lock:
            if not session.running:
                break
                
        # 1. Grab snapshot of latest frame
        with session._grab_lock:
            frame = session._latest_raw_frame
            
        if frame is None:
            time.sleep(0.01)
            continue
            
        # 2. Grab latest inference results
        with session._infer_lock:
            boxes_np, confs_np, face_lms_smoothed, raw_centers, p_count = session._latest_infer_res
            
        # 3. Form annotated frame
        annotated = frame.copy() if len(boxes_np) > 0 else frame
        if len(boxes_np) > 0:
            annotated = _draw_lightweight_boxes(annotated, boxes_np, confs_np)
            
        # 4. Form blurred frame conditionally
        blurred_frame = annotated.copy()
        orig_h, orig_w = blurred_frame.shape[:2]
        
        # Build a single binary mask for all areas to anonymize
        anon_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        covered_face_areas = []
        
        # Generate Convex Hull mask for each face and add it to anon_mask
        has_small_face = False
        for lms in face_lms_smoothed:
            if len(lms) == 0:
                continue
            pts = np.array(lms, dtype=np.int32)
            hull = cv2.convexHull(pts)
            cv2.fillConvexPoly(anon_mask, hull, 255)
            
            x, y, w, h = cv2.boundingRect(pts)
            covered_face_areas.append((max(0, x), max(0, y), min(orig_w, x + w), min(orig_h, y + h)))
            
            # Check for small face to apply heavier dilation
            face_area = w * h
            if face_area < (orig_w * orig_h * 0.015):
                has_small_face = True
                
        # Morphological Expansion (Dilation) to cover edges cleanly
        if len(face_lms_smoothed) > 0:
            k_size = 35 if has_small_face else 20
            kernel = np.ones((k_size, k_size), np.uint8)
            anon_mask = cv2.dilate(anon_mask, kernel, iterations=1)
                
        # Check YOLO boxes for fallbacks (Top 45% of person if no face mesh overlaps)
        for pb in boxes_np:
            px1, py1, px2, py2 = map(int, pb)
            px1, py1 = max(0, px1), max(0, py1)
            px2, py2 = min(orig_w, px2), min(orig_h, py2)
            
            p_width = px2 - px1
            p_height = py2 - py1
            if p_width <= 0 or p_height <= 0:
                continue
                
            hx1 = px1
            hy1 = py1
            hx2 = px2
            hy2 = py1 + int(p_height * 0.45)
            
            is_covered = False
            for (fx1, fy1, fx2, fy2) in covered_face_areas:
                ix1 = max(hx1, fx1)
                iy1 = max(hy1, fy1)
                ix2 = min(hx2, fx2)
                iy2 = min(hy2, fy2)
                if ix2 > ix1 and iy2 > iy1:
                    is_covered = True
                    break
                    
            if not is_covered:
                cv2.rectangle(anon_mask, (hx1, hy1), (hx2, hy2), 255, -1)
                
        # Apply heavy pixelation where anon_mask > 0
        if cv2.countNonZero(anon_mask) > 0:
            scale_factor = max(1, orig_w // 12)  # target roughly 12px blocks
            small_w, small_h = orig_w // scale_factor, orig_h // scale_factor
            if small_w > 0 and small_h > 0:
                small = cv2.resize(blurred_frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
                pixelated = cv2.resize(small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                
                mask_bool = anon_mask > 0
                blurred_frame[mask_bool] = pixelated[mask_bool]
        # 5. Output spatial data
        spatial = session._spatial
        spatial.offer_frame(frame)
        spatial.update(raw_centers)

        # 6. Encode JPEGs
        jpeg_raw = _frame_to_jpeg(frame)
        jpeg_ann = _frame_to_jpeg(annotated)
        jpeg_blur = _frame_to_jpeg(blurred_frame)
        
        with session.lock:
            session.latest_jpeg = jpeg_raw
            session.latest_annotated_jpeg = jpeg_ann
            session.latest_blurred_jpeg = jpeg_blur
            
        with session.stream_event:
            session.stream_event.notify_all()
            
        frame_count += 1
        now = time.time()
        if now - t0 >= 5.0:
            session.fps_metrics["stream"] = round(frame_count / (now - t0), 1)
            frame_count = 0
            t0 = now
            
        # Loop at ~30 FPS constant
        time.sleep(0.033)

    print(f"{tag} Stream worker stopped.")


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

    def get_multimodal_snapshot(self) -> dict:
        """Return fused data from all three multi-modal layers."""
        sms = sms_gateway.get_sms_stats()

        # Aggregate total vision count
        total_vision = 0
        with self._lock:
            sessions = list(self._sessions.values())
        for s in sessions:
            with s.lock:
                total_vision += s.metrics.get("people", 0)

        return {
            "signal": sig,
            "acoustic": aco,
            "sms": sms,
            "vision_count": total_vision,
            "ts": time.time(),
        }

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
        for t in (session._grabber_thread, session._inference_thread, session._stream_thread):
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
            if hasattr(session, '_face_tracker'):
                session._face_tracker.reset()

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

        # Launch streamer thread (output at 30 FPS)
        st = threading.Thread(
            target=_stream_loop,
            args=(session,),
            daemon=True,
            name=f"sentinel-stream-{camera_id}"
        )
        st.start()

        with session.lock:
            session._grabber_thread = gt
            session._inference_thread = it
            session._stream_thread = st

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
        return {"people": 0, "risk": "LOW", "safety": "SAFE"}
    session = _manager.get_session(cid)
    if session is None:
        return {"people": 0, "risk": "LOW", "safety": "SAFE"}
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
