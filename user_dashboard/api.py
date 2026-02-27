from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
import sys
import os
import json
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared import auth, db
from shared import jwt_utils
from user_dashboard import monitor
from fastapi import File, UploadFile
from fastapi.responses import FileResponse
import shutil

app = FastAPI(title="Sentinel React API")

# CORS — dev only, restrict origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Seed admin on startup ───────────────────────────────────────────────────

@app.on_event("startup")
def on_startup():
    auth.seed_admin()

# ── Request Models ──────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    password: str

class MonitorStartRequest(BaseModel):
    source: str = "0"           # "0" = webcam, RTSP URL, IP cam URL
    model: str = "yolov8n-pose.pt"

class CameraAddRequest(BaseModel):
    source_url: str
    label: str = ""

# ── Auth Endpoints ──────────────────────────────────────────────────────────

@app.post("/api/auth/login")
def login(req: LoginRequest):
    """Verify credentials, return JWT + role for any valid user."""
    if not auth.verify_user(req.username, req.password):
        raise HTTPException(status_code=401, detail="Invalid credentials.")
    role = auth.get_user_role(req.username)
    if role is None:
        raise HTTPException(status_code=401, detail="User not found.")
    token = jwt_utils.encode_token(req.username, role)
    return {"success": True, "token": token, "username": req.username, "role": role}


@app.post("/api/auth/register")
def register(req: RegisterRequest):
    """Register a public user (role = 'user'). Cannot create admins."""
    ok, err = auth.register_user(req.username, req.password)
    if not ok:
        raise HTTPException(status_code=400, detail=err)
    return {"success": True, "message": "Account created. You can now sign in."}


# ══════════════════════════════════════════════════════════════════════════
#  Multi-Camera Endpoints
# ══════════════════════════════════════════════════════════════════════════

@app.post("/api/cameras/add")
def cameras_add(req: CameraAddRequest):
    """Add a new camera session. Returns camera_id."""
    mgr = monitor.get_manager()
    camera_id, err = mgr.add_camera(req.source_url, req.label)
    if err:
        raise HTTPException(status_code=409, detail=err)
    return {"success": True, "camera_id": camera_id}


@app.post("/api/cameras/remove/{camera_id}")
def cameras_remove(camera_id: str):
    """Stop and remove a camera session."""
    mgr = monitor.get_manager()
    ok, err = mgr.remove_camera(camera_id)
    if not ok:
        raise HTTPException(status_code=404, detail=err)
    return {"success": True, "message": "Camera removed."}


@app.post("/api/cameras/start/{camera_id}")
def cameras_start(camera_id: str):
    """Start detection on a specific camera."""
    mgr = monitor.get_manager()
    ok, err = mgr.start_camera(camera_id)
    if not ok:
        raise HTTPException(status_code=400, detail=err)
    return {"success": True, "message": "Camera started."}


@app.post("/api/cameras/stop/{camera_id}")
def cameras_stop(camera_id: str):
    """Stop detection on a specific camera."""
    mgr = monitor.get_manager()
    ok, err = mgr.stop_camera(camera_id)
    if not ok:
        raise HTTPException(status_code=400, detail=err)
    return {"success": True, "message": "Camera stopped."}


@app.get("/api/cameras/list")
def cameras_list():
    """Returns array of all camera sessions with status."""
    mgr = monitor.get_manager()
    return {"cameras": mgr.list_sessions()}


@app.get("/api/cameras/stream/admin/{camera_id}")
def cameras_stream_admin(camera_id: str):
    """MJPEG admin stream for a specific camera (no blur)."""
    mgr = monitor.get_manager()
    session = mgr.get_session(camera_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Camera not found.")

    def _generate():
        while True:
            frame = mgr.get_frame(camera_id, stream_type="admin")
            if not frame:
                time.sleep(0.01)
                continue
                
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                frame +
                b"\r\n"
            )
            time.sleep(0.033)  # ~30 fps

    return StreamingResponse(
        _generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/api/cameras/stream/public/{camera_id}")
def cameras_stream_public(camera_id: str):
    """MJPEG public stream for a specific camera (with face blur)."""
    mgr = monitor.get_manager()
    session = mgr.get_session(camera_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Camera not found.")

    def _generate():
        while True:
            frame = mgr.get_frame(camera_id, stream_type="public")
            if not frame:
                time.sleep(0.01)
                continue
                
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                frame +
                b"\r\n"
            )
            time.sleep(0.033)  # ~30 fps

    return StreamingResponse(
        _generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/api/cameras/stream/{camera_id}")
def cameras_stream(camera_id: str):
    """MJPEG stream for a specific camera (defaults to admin for backward compatibility)."""
    mgr = monitor.get_manager()
    session = mgr.get_session(camera_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Camera not found.")

    def _generate():
        while True:
            frame = mgr.get_frame(camera_id, stream_type="admin")
            if not frame:
                time.sleep(0.01)
                continue
                

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                frame +
                b"\r\n"
            )
            time.sleep(0.033)  # ~30 fps

    return StreamingResponse(
        _generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/cameras/spatial/{camera_id}")
def cameras_spatial(camera_id: str):
    """SSE spatial projection stream for a specific camera."""
    mgr = monitor.get_manager()
    session = mgr.get_session(camera_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Camera not found.")

    def _generate():
        last_ts = 0
        while True:
            data = mgr.get_spatial_data(camera_id)
            if data["ts"] > last_ts:
                payload = json.dumps(data)
                yield f"data: {payload}\n\n"
                last_ts = data["ts"]
            time.sleep(0.20)

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/rooms/spatial/{room_name}")
def rooms_spatial(room_name: str):
    """SSE spatial projection stream for all cameras in a room fused together."""
    mgr = monitor.get_manager()
    
    # Optional basic room validation, though mgr.get_room_spatial_data is safe
    if room_name not in ["Room 1", "Room 2"]:
        raise HTTPException(status_code=404, detail="Room not found.")

    def _generate():
        last_ts = 0
        while True:
            data = mgr.get_room_spatial_data(room_name)
            # Only yield if it's successfully returned a valid grid and has advanced in time
            if data["ready"] and data["ts"] > last_ts:
                payload = json.dumps(data)
                yield f"data: {payload}\n\n"
                last_ts = data["ts"]
            time.sleep(0.20)

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

@app.get("/api/public/heatmap/stream")
def public_heatmap_stream(room: str = "Room 1"):
    """Public read-only SSE spatial projection stream for the active room layout."""
    # Mirror the exact same behavior as rooms_spatial without altering backend scope
    mgr = monitor.get_manager()
    
    if room not in ["Room 1", "Room 2"]:
        raise HTTPException(status_code=404, detail="Room not found.")

    def _generate():
        last_ts = 0
        while True:
            data = mgr.get_room_spatial_data(room)
            if data["ready"] and data["ts"] > last_ts:
                payload = json.dumps(data)
                yield f"data: {payload}\n\n"
                last_ts = data["ts"]
            time.sleep(0.20)

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

class CalibrationRequest(BaseModel):
    camera_pts: list[list[float]]
    blueprint_pts: list[list[float]]
    bp_w: int
    bp_h: int

@app.post("/api/cameras/spatial/calibrate/{camera_id}")
def cameras_spatial_calibrate(camera_id: str, req: CalibrationRequest):
    """Set the real-world floor boundary for perspective projection via homography."""
    mgr = monitor.get_manager()
    ok, err = mgr.calibrate_spatial(camera_id, req.camera_pts, req.blueprint_pts, req.bp_w, req.bp_h)
    if not ok:
        raise HTTPException(status_code=400, detail=err)
    return {"success": True, "message": "Calibration matrix applied."}

@app.post("/api/cameras/spatial/blueprint/{camera_id}")
async def upload_blueprint(camera_id: str, file: UploadFile = File(...)):
    """Upload a blueprint image to use for spatial homography mapping."""
    import os
    blueprint_dir = os.path.join(os.path.dirname(__file__), "blueprints")
    os.makedirs(blueprint_dir, exist_ok=True)
    file_path = os.path.join(blueprint_dir, f"{camera_id}.jpg")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return {"success": True, "message": "Blueprint uploaded successfully"}

@app.get("/api/cameras/spatial/blueprint/{camera_id}")
def get_blueprint(camera_id: str):
    """Fetch the uploaded blueprint image for the given camera."""
    import os
    blueprint_dir = os.path.join(os.path.dirname(__file__), "blueprints")
    file_path = os.path.join(blueprint_dir, f"{camera_id}.jpg")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Blueprint not found.")
    return FileResponse(file_path)


@app.post("/api/cameras/start-all")
def cameras_start_all():
    """Start all stopped cameras."""
    mgr = monitor.get_manager()
    count = mgr.start_all()
    return {"success": True, "started": count}


@app.post("/api/cameras/stop-all")
def cameras_stop_all():
    """Stop all running cameras."""
    mgr = monitor.get_manager()
    count = mgr.stop_all()
    return {"success": True, "stopped": count}


# ══════════════════════════════════════════════════════════════════════════
#  AI Tactical Recommendations
# ══════════════════════════════════════════════════════════════════════════

from user_dashboard import ai_engine

@app.get("/api/ai/recommendations")
def ai_recommendations():
    """Return AI tactical recommendations based on live camera data."""
    mgr = monitor.get_manager()
    result = ai_engine.get_recommendations(mgr)
    return result


# ══════════════════════════════════════════════════════════════════════════
#  Legacy Live Monitor Endpoints (backward compatible)
# ══════════════════════════════════════════════════════════════════════════

@app.post("/api/monitor/start")
def monitor_start(req: MonitorStartRequest):
    """Start the YOLO detection loop in a background thread."""
    if monitor.is_running():
        return {"success": False, "message": "Already running."}
    started = monitor.start(source=req.source, model_name=req.model)
    return {"success": started, "message": "Monitor started." if started else "Failed to start."}


@app.post("/api/monitor/stop")
def monitor_stop():
    """Stop the detection loop and release camera/audio resources."""
    stopped = monitor.stop()
    return {"success": stopped, "message": "Monitor stopped." if stopped else "Not running."}


@app.get("/api/monitor/status")
def monitor_status():
    """Returns current running state etc."""
    mgr = monitor.get_manager()
    cameras = mgr.list_sessions()
    return {
        "running": monitor.is_running(),
        "camera_count": len(cameras),
        "cameras": cameras,
        **monitor.get_metrics()
    }


@app.get("/api/monitor/stream")
def monitor_stream():
    """MJPEG stream — each frame sent as multipart/x-mixed-replace.
    Auto-starts monitor if not running."""
    import time as _time

    # Auto-start if not running
    if not monitor.is_running():
        monitor.start(source="0", model_name="yolov8n-pose.pt")
        # Brief wait for first frame
        deadline = _time.time() + 2.0
        while _time.time() < deadline:
            jpeg = monitor.get_latest_jpeg()
            if jpeg and len(jpeg) > 5000:
                break
            _time.sleep(0.1)

    def _generate():
        while True:
            jpeg = monitor.get_latest_jpeg()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                jpeg +
                b"\r\n"
            )
            time.sleep(0.033)  # ~30 fps

    return StreamingResponse(
        _generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ── Data Endpoints ──────────────────────────────────────────────────────────

@app.get("/api/data/latest")
def get_latest():
    database = db.get_db()
    latest = database.get_latest_event()
    if not latest:
        return {"available": False}
    return {
        "available": True,
        "risk_level": latest.get("risk_level", "LOW"),
        "people_count": latest.get("people_count", 0),
        "timestamp": latest.get("timestamp").isoformat() if latest.get("timestamp") else None
    }

@app.get("/api/data/history")
def get_history(limit: int = 50):
    database = db.get_db()
    history = database.get_recent_history(limit=limit)
    if not history:
        return []
    return [
        {
            "timestamp": h.get("timestamp").isoformat() if h.get("timestamp") else None,
            "people_count": h.get("people_count", 0)
        }
        for h in history
    ]

# ── Spatial Density Stream ──────────────────────────────────────────────────

@app.get("/api/spatial/stream")
def spatial_stream():
    """SSE stream — automatic spatial projection at ~5 Hz."""
    def _generate():
        last_ts = 0
        while True:
            data = monitor.get_spatial_data()
            if data["ts"] > last_ts:
                payload = json.dumps(data)
                yield f"data: {payload}\n\n"
                last_ts = data["ts"]
            time.sleep(0.20)  # ~5 Hz

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )





@app.get("/api/monitor/snapshot")
def monitor_snapshot():
    """Return a single JPEG frame. Auto-starts monitor if not running."""
    # Auto-start if not running
    if not monitor.is_running():
        monitor.start(source="0", model_name="yolov8n-pose.pt")

    # Wait up to 2 seconds for a real frame
    import time as _time
    deadline = _time.time() + 2.0
    jpeg = None
    while _time.time() < deadline:
        jpeg = monitor.get_latest_jpeg()
        # _blank_frame returns a small placeholder — check we have a real frame
        if jpeg and len(jpeg) > 5000:  # real JPEG > 5KB
            break
        _time.sleep(0.1)

    if not jpeg or len(jpeg) <= 5000:
        raise HTTPException(503, "Camera not ready. Please try again.")

    return Response(content=jpeg, media_type="image/jpeg")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
