from fastapi import FastAPI, HTTPException, UploadFile, File, Form
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
from user_dashboard import calibration as calib_mod

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


# ── Live Monitor Endpoints ──────────────────────────────────────────────────

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
    """Returns current running state and latest metrics."""
    return {
        "running": monitor.is_running(),
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
            time.sleep(0.04)  # ~25 fps

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

# ── Calibration Endpoints ───────────────────────────────────────────────────

@app.post("/api/calibration/save")
async def save_calibration(
    blueprint: UploadFile = File(...),
    camera_pts: str = Form(...),
    floor_pts: str = Form(...),
    area_type: str = Form("Open Ground"),
):
    """Save calibration: blueprint image + 4+4 points → compute homography."""
    try:
        cam = json.loads(camera_pts)
        flr = json.loads(floor_pts)
    except json.JSONDecodeError:
        raise HTTPException(400, "camera_pts and floor_pts must be valid JSON arrays.")

    bp_bytes = await blueprint.read()
    if len(bp_bytes) > 10 * 1024 * 1024:
        raise HTTPException(400, "Blueprint file too large (max 10MB).")

    snap, err = calib_mod.save_calibration(
        camera_pts=cam,
        floor_pts=flr,
        area_type=area_type,
        blueprint_bytes=bp_bytes,
        blueprint_filename=blueprint.filename or "blueprint.png",
    )
    if snap is None:
        raise HTTPException(400, err)

    # Atomic swap into running monitor
    monitor.set_calib(snap)
    return {"success": True, "message": "Calibration saved and activated."}


@app.get("/api/calibration/status")
def calibration_status():
    """Check if calibration is active."""
    calib = monitor.get_calib()
    if calib is None:
        return {"calibrated": False}
    return {
        "calibrated": True,
        "area_type": calib.area_type,
        "bp_width": calib.bp_width,
        "bp_height": calib.bp_height,
        "blueprint_path": calib.blueprint_path,
    }


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


@app.get("/api/heatmap/stream")
def heatmap_stream():
    """SSE stream — heatmap positions + density grid at ~3 Hz."""
    def _generate():
        last_ts = 0
        while True:
            data = monitor.get_heatmap_data()
            if data["ts"] > last_ts:
                payload = json.dumps(data)
                yield f"data: {payload}\n\n"
                last_ts = data["ts"]
            time.sleep(0.33)  # ~3 Hz

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/calibration/blueprint")
def serve_blueprint():
    """Serve the active blueprint image."""
    calib = monitor.get_calib()
    if calib is None or not os.path.exists(calib.blueprint_path):
        raise HTTPException(404, "No calibration or blueprint found.")
    with open(calib.blueprint_path, 'rb') as f:
        data = f.read()
    ext = os.path.splitext(calib.blueprint_path)[1].lower()
    ct = {
        '.png': 'image/png', '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg', '.svg': 'image/svg+xml',
    }.get(ext, 'image/png')
    return Response(content=data, media_type=ct)


@app.get("/api/calibration/current")
def calibration_current():
    """Return full calibration state for frontend restore on mount."""
    calib = monitor.get_calib()
    if calib is None:
        return {"calibrated": False}

    # Read the persisted JSON to get the original points
    _, meta = calib_mod.load_calibration()
    return {
        "calibrated": True,
        "area_type": calib.area_type,
        "bp_width": calib.bp_width,
        "bp_height": calib.bp_height,
        "blueprint_url": f"/api/calibration/blueprint",
        "camera_pts": meta.get("camera_pts", []) if meta else [],
        "floor_pts": meta.get("floor_pts", []) if meta else [],
    }


@app.delete("/api/calibration/reset")
def calibration_reset():
    """Clear calibration from disk and monitor."""
    # Remove persisted file
    active_path = calib_mod.ACTIVE_FILE
    if os.path.exists(active_path):
        os.remove(active_path)
    # Clear from running monitor
    monitor.set_calib(None)
    return {"success": True, "message": "Calibration reset."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

