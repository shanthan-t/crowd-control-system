"""
spatial_engine.py — Automatic Bird's-Eye Spatial Intelligence Engine

Manual calibration using 4 points provided via API.
Projects detections into a structured top-view density grid using homography.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ── Spatial configuration ──────────────────────────────────────────────────
TOP_VIEW_W = 800
TOP_VIEW_H = 600
GRID_W = 40
GRID_H = 30
BLUR_SIGMA = 1.2

@dataclass
class SpatialSnapshot:
    grid: np.ndarray
    points: np.ndarray
    width: int
    height: int
    timestamp: float
    ready: bool

class SpatialEngine:
    """Manual bird's-eye spatial engine driven by a provided homography quad."""

    def __init__(
        self,
        grid_w: int = GRID_W,
        grid_h: int = GRID_H,
        top_w: int = TOP_VIEW_W,
        top_h: int = TOP_VIEW_H,
    ):
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.top_w = top_w
        self.top_h = top_h

        # Geometry
        self.camera_quad: Optional[np.ndarray] = None
        self.H: Optional[np.ndarray] = None

        # Live spatial outputs
        self.grid = np.zeros((grid_h, grid_w), dtype=np.float32)
        self.projected_points = np.empty((0, 2), dtype=np.float32)
        self.timestamp = 0.0

        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self.H is not None

    def offer_frame(self, frame_bgr: np.ndarray) -> None:
        """No longer constructs a background model. Just a stub for compatibility."""
        pass

    def set_calibration(self, camera_pts: List[List[float]], blueprint_pts: List[List[float]], bp_w: int, bp_h: int) -> bool:
        """Receive 4 camera points and 4 blueprint points to compute homography."""
        if len(camera_pts) != 4 or len(blueprint_pts) != 4:
            return False
            
        try:
            cam_quad = np.array(camera_pts, dtype=np.float32)
            bp_quad = np.array(blueprint_pts, dtype=np.float32)
            
            H, _ = cv2.findHomography(cam_quad, bp_quad)
            if H is None:
                return False
                
            with self._lock:
                self.H = H
                self.top_w = bp_w
                self.top_h = bp_h
                # Optional: adjust density grid resolution based on aspect ratio
                # E.g., keep width=40, scale height, or just use 40x30 default
                self.grid_w = GRID_W
                self.grid_h = max(10, int((bp_h / bp_w) * GRID_W)) if bp_w > 0 else GRID_H
                self.grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
            return True
        except Exception as e:
            print(f"Error computing calibration homography: {e}")
            return False

    def update(self, raw_centers: List[List[float]]) -> None:
        """Project detections into top-view and update the density grid."""
        now = time.time()
        with self._lock:
            H = self.H
            
        if not raw_centers or H is None:
            with self._lock:
                self.grid[:] = 0.0
                self.projected_points = np.empty((0, 2), dtype=np.float32)
                self.timestamp = now
            return

        pts = np.array(raw_centers, dtype=np.float32).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(pts, H).reshape(-1, 2)

        # Filter points within bounds
        mask = (
            (proj[:, 0] >= 0) & (proj[:, 0] < self.top_w) &
            (proj[:, 1] >= 0) & (proj[:, 1] < self.top_h)
        )
        proj = proj[mask]

        # Build density grid
        grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        if len(proj) > 0:
            xs = (proj[:, 0] / self.top_w * self.grid_w).astype(np.int32)
            ys = (proj[:, 1] / self.top_h * self.grid_h).astype(np.int32)
            xs = np.clip(xs, 0, self.grid_w - 1)
            ys = np.clip(ys, 0, self.grid_h - 1)
            np.add.at(grid, (ys, xs), 1.0)
            grid = cv2.GaussianBlur(grid, (0, 0), sigmaX=BLUR_SIGMA, sigmaY=BLUR_SIGMA)
            maxv = float(grid.max())
            if maxv > 0:
                grid /= maxv

        with self._lock:
            self.grid = grid
            self.projected_points = proj
            self.timestamp = now

    def get_snapshot(self) -> SpatialSnapshot:
        """Thread-safe read of the latest spatial snapshot."""
        with self._lock:
            return SpatialSnapshot(
                grid=self.grid.copy(),
                points=self.projected_points.copy(),
                width=self.top_w,
                height=self.top_h,
                timestamp=self.timestamp,
                ready=(self.H is not None)
            )

    def reset(self) -> None:
        with self._lock:
            self.grid[:] = 0.0
            self.projected_points = np.empty((0, 2), dtype=np.float32)
            self.timestamp = 0.0
            self.camera_quad = None
            self.H = None

def extract_bbox_centers(boxes_xyxy: np.ndarray) -> List[List[float]]:
    """
    Extract bottom-center points from bounding boxes.

    Input:  Nx4 array [[x1, y1, x2, y2], ...]
    Output: [[cx, cy], ...] where cx = midpoint, cy = bottom edge
    """
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return []
    centers = []
    for box in boxes_xyxy:
        x1, y1, x2, y2 = box[:4]
        cx = (x1 + x2) / 2.0
        cy = float(y2)
        centers.append([cx, cy])
    return centers
