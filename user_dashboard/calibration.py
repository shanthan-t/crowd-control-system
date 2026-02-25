"""
calibration.py — Floor Plan Calibration & Homography Module
Computes, stores, and applies perspective transforms from
camera space → blueprint space for real-time crowd heatmaps.
"""

import cv2
import numpy as np
import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Optional, List, Tuple
from PIL import Image

# ── Constants ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CALIB_DIR = os.path.join(BASE_DIR, 'calibrations')
BLUEPRINT_DIR = os.path.join(CALIB_DIR, 'blueprints')
ACTIVE_FILE = os.path.join(CALIB_DIR, 'active.json')

GRID_COLS = 20
GRID_ROWS = 12


# ── Frozen snapshot — thread-safe, immutable ───────────────────────────────

@dataclass(frozen=True)
class CalibSnapshot:
    """Immutable calibration config read by the detection thread.
    Swap atomically via single pointer assignment under lock."""
    H: object               # 3×3 np.float64 homography matrix
    camera_quad: object      # 4×2 np.float32 camera quadrilateral
    bp_width: int
    bp_height: int
    blueprint_path: str
    area_type: str


# ── Persistence ────────────────────────────────────────────────────────────

def save_calibration(
    camera_pts: List[List[float]],
    floor_pts: List[List[float]],
    area_type: str,
    blueprint_bytes: bytes,
    blueprint_filename: str,
    frame_width: int = 640,
    frame_height: int = 480,
) -> Tuple[Optional[CalibSnapshot], str]:
    """Compute homography, persist config + blueprint, return snapshot.

    camera_pts and floor_pts arrive as percentages (0-100) of their
    respective containers. We convert to actual pixel coords here.

    Returns:
        (CalibSnapshot, "") on success
        (None, error_message) on failure
    """
    # Validate point counts
    if len(camera_pts) != 4 or len(floor_pts) != 4:
        return None, "Exactly 4 camera points and 4 floor points required."

    # Save blueprint file FIRST so we can read its dimensions
    os.makedirs(BLUEPRINT_DIR, exist_ok=True)
    ext = os.path.splitext(blueprint_filename)[1] or '.png'
    bp_name = f"{uuid.uuid4().hex[:12]}{ext}"
    bp_path = os.path.join(BLUEPRINT_DIR, bp_name)
    with open(bp_path, 'wb') as f:
        f.write(blueprint_bytes)

    # Get blueprint dimensions
    try:
        img = Image.open(bp_path)
        bp_w, bp_h = img.size
        img.close()
    except Exception:
        bp_w, bp_h = 1200, 800  # fallback

    # Convert percentage coords → actual pixel coords
    cam_px = [[p[0] / 100.0 * frame_width, p[1] / 100.0 * frame_height] for p in camera_pts]
    flr_px = [[p[0] / 100.0 * bp_w, p[1] / 100.0 * bp_h] for p in floor_pts]

    src = np.float32(cam_px)
    dst = np.float32(flr_px)

    # Compute homography
    H, mask = cv2.findHomography(src, dst)
    if H is None:
        return None, "Homography computation failed — points may be collinear."

    # Persist calibration JSON
    os.makedirs(CALIB_DIR, exist_ok=True)
    data = {
        "camera_pts": cam_px,     # actual pixel coords (not %)
        "floor_pts": flr_px,       # actual pixel coords (not %)
        "area_type": area_type,
        "blueprint_path": bp_path,
        "blueprint_width": bp_w,
        "blueprint_height": bp_h,
        "homography": H.tolist(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(ACTIVE_FILE, 'w') as f:
        json.dump(data, f, indent=2)

    # Build snapshot
    snap = CalibSnapshot(
        H=np.float64(H),
        camera_quad=np.float32(cam_px),  # actual pixel coords for polygon test
        bp_width=bp_w,
        bp_height=bp_h,
        blueprint_path=bp_path,
        area_type=area_type,
    )
    return snap, ""


def load_calibration() -> Tuple[Optional[CalibSnapshot], Optional[dict]]:
    """Load persisted calibration from disk.

    Returns:
        (CalibSnapshot, metadata_dict) if exists
        (None, None) if not calibrated
    """
    if not os.path.exists(ACTIVE_FILE):
        return None, None
    try:
        with open(ACTIVE_FILE) as f:
            data = json.load(f)
        H = np.float64(data["homography"])
        snap = CalibSnapshot(
            H=H,
            camera_quad=np.float32(data["camera_pts"]),
            bp_width=data.get("blueprint_width", 1200),
            bp_height=data.get("blueprint_height", 800),
            blueprint_path=data.get("blueprint_path", ""),
            area_type=data.get("area_type", ""),
        )
        return snap, data
    except Exception as e:
        print(f"[Calibration] Load error: {e}")
        return None, None


# ── Core transforms ────────────────────────────────────────────────────────

def filter_inside_quad(
    points: List[List[float]],
    quad: np.ndarray,
) -> List[List[float]]:
    """Keep only points that fall inside the camera quadrilateral."""
    if len(points) == 0:
        return []
    contour = quad.reshape(-1, 1, 2).astype(np.float32)
    return [
        p for p in points
        if cv2.pointPolygonTest(contour, (float(p[0]), float(p[1])), False) >= 0
    ]


def transform_points(
    points: List[List[float]],
    H: np.ndarray,
) -> List[List[float]]:
    """Apply homography H to Nx2 points → blueprint coordinates."""
    if len(points) == 0:
        return []
    pts = np.float32(points).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts, H)
    return transformed.reshape(-1, 2).tolist()


def build_density_grid(
    positions: List[List[float]],
    bp_w: int,
    bp_h: int,
    grid_w: int = GRID_COLS,
    grid_h: int = GRID_ROWS,
) -> List[List[int]]:
    """Bin transformed positions into a grid_w × grid_h density matrix."""
    grid = [[0] * grid_w for _ in range(grid_h)]
    if not positions or bp_w <= 0 or bp_h <= 0:
        return grid
    for (x, y) in positions:
        col = int(x / bp_w * grid_w)
        row = int(y / bp_h * grid_h)
        col = max(0, min(grid_w - 1, col))
        row = max(0, min(grid_h - 1, row))
        grid[row][col] += 1
    return grid


def extract_bbox_centers(boxes_xyxy: np.ndarray) -> List[List[float]]:
    """Extract bottom-center (feet position) from YOLO xyxy boxes.

    Args:
        boxes_xyxy: Nx4 array of [x1, y1, x2, y2]
    Returns:
        List of [cx, cy] where cy = bottom edge (feet)
    """
    centers = []
    for box in boxes_xyxy:
        cx = float((box[0] + box[2]) / 2)
        cy = float(box[3])  # bottom = feet
        centers.append([cx, cy])
    return centers
