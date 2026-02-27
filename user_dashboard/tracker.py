"""
tracker.py — Lightweight Centroid Tracker with EMA Smoothing
=============================================================

Simple IoU/distance-based tracker that:
  1. Matches new detections to existing tracks via centroid distance
  2. Applies exponential moving average (EMA) to bounding box coordinates
  3. Keeps tracks alive for up to MAX_DISAPPEARED frames before dropping
  4. Provides stable person IDs and smooth bounding boxes

No external dependencies — uses only numpy and scipy.spatial.
"""

from __future__ import annotations

import numpy as np
from collections import OrderedDict
from typing import List, Tuple, Optional


# ── Configuration ─────────────────────────────────────────────────────────
MAX_DISAPPEARED = 8      # frames before a track is dropped
MATCH_RADIUS    = 120.0  # max centroid distance (pixels) to match
EMA_ALPHA       = 0.4    # EMA weight for new detections (lower = smoother)


class TrackedObject:
    """Single tracked person with EMA-smoothed bounding box."""

    __slots__ = ('object_id', 'bbox', 'centroid', 'disappeared', 'confidence')

    def __init__(self, object_id: int, bbox: np.ndarray, confidence: float = 0.5):
        self.object_id = object_id
        self.bbox = bbox.astype(np.float64)        # [x1, y1, x2, y2] smoothed
        self.centroid = self._compute_centroid(bbox)
        self.disappeared = 0
        self.confidence = confidence

    @staticmethod
    def _compute_centroid(bbox: np.ndarray) -> np.ndarray:
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        return np.array([cx, cy], dtype=np.float64)

    def update(self, bbox: np.ndarray, confidence: float = 0.5) -> None:
        """EMA update of bounding box coordinates."""
        new_bbox = bbox.astype(np.float64)
        self.bbox = EMA_ALPHA * new_bbox + (1.0 - EMA_ALPHA) * self.bbox
        self.centroid = self._compute_centroid(self.bbox)
        self.disappeared = 0
        self.confidence = confidence

    def mark_disappeared(self) -> None:
        self.disappeared += 1

    @property
    def bottom_center(self) -> Tuple[float, float]:
        """Return smoothed bottom-center (feet position)."""
        cx = (self.bbox[0] + self.bbox[2]) / 2.0
        cy = self.bbox[3]  # bottom edge
        return (float(cx), float(cy))


class CentroidTracker:
    """
    Lightweight multi-object tracker.

    Algorithm:
      1. If no existing tracks → register all detections as new tracks
      2. If no new detections → mark all tracks as disappeared
      3. Otherwise → compute pairwise centroid distances
         a. Greedily match closest pairs under MATCH_RADIUS
         b. Unmatched detections → register as new tracks
         c. Unmatched tracks → mark disappeared, drop after MAX_DISAPPEARED

    Returns stable bounding boxes suitable for heatmap generation.
    """

    def __init__(self,
                 max_disappeared: int = MAX_DISAPPEARED,
                 match_radius: float = MATCH_RADIUS):
        self._next_id = 0
        self._tracks: OrderedDict[int, TrackedObject] = OrderedDict()
        self._max_disappeared = max_disappeared
        self._match_radius = match_radius

    @property
    def tracks(self) -> OrderedDict[int, TrackedObject]:
        return self._tracks

    @property
    def active_count(self) -> int:
        """Number of active (not disappeared) tracks."""
        return sum(1 for t in self._tracks.values() if t.disappeared == 0)

    @property
    def total_count(self) -> int:
        """Total tracked objects including recently disappeared."""
        return len(self._tracks)

    def _register(self, bbox: np.ndarray, confidence: float = 0.5) -> TrackedObject:
        t = TrackedObject(self._next_id, bbox, confidence)
        self._tracks[self._next_id] = t
        self._next_id += 1
        return t

    def _deregister(self, object_id: int) -> None:
        del self._tracks[object_id]

    def update(self, boxes: np.ndarray,
               confidences: Optional[np.ndarray] = None) -> List[TrackedObject]:
        """
        Update tracker with new detections.

        Args:
            boxes: Nx4 array [[x1, y1, x2, y2], ...]
            confidences: N-length array of confidence scores

        Returns:
            List of all active TrackedObjects (including recently disappeared)
        """
        if boxes is None or len(boxes) == 0:
            # No detections — mark all as disappeared
            to_remove = []
            for oid, track in self._tracks.items():
                track.mark_disappeared()
                if track.disappeared > self._max_disappeared:
                    to_remove.append(oid)
            for oid in to_remove:
                self._deregister(oid)
            return list(self._tracks.values())

        boxes = np.atleast_2d(boxes).astype(np.float64)
        n_det = len(boxes)
        confs = confidences if confidences is not None else np.full(n_det, 0.5)

        # No existing tracks → register all
        if len(self._tracks) == 0:
            for i in range(n_det):
                self._register(boxes[i], float(confs[i]))
            return list(self._tracks.values())

        # Compute centroids of new detections
        det_centroids = np.zeros((n_det, 2), dtype=np.float64)
        for i, box in enumerate(boxes):
            det_centroids[i, 0] = (box[0] + box[2]) / 2.0
            det_centroids[i, 1] = (box[1] + box[3]) / 2.0

        # Existing track centroids
        track_ids = list(self._tracks.keys())
        track_centroids = np.array(
            [self._tracks[tid].centroid for tid in track_ids],
            dtype=np.float64
        )

        # Pairwise distance matrix
        n_tracks = len(track_ids)
        dist = np.zeros((n_tracks, n_det), dtype=np.float64)
        for i in range(n_tracks):
            for j in range(n_det):
                diff = track_centroids[i] - det_centroids[j]
                dist[i, j] = np.sqrt(diff[0]**2 + diff[1]**2)

        # Greedy matching (sorted by distance)
        matched_tracks = set()
        matched_dets = set()

        # Sort all pairs by distance
        flat_indices = np.argsort(dist, axis=None)
        for flat_idx in flat_indices:
            i = flat_idx // n_det
            j = flat_idx % n_det

            if i in matched_tracks or j in matched_dets:
                continue
            if dist[i, j] > self._match_radius:
                break  # all remaining are too far

            # Match found — EMA update
            tid = track_ids[i]
            self._tracks[tid].update(boxes[j], float(confs[j]))
            matched_tracks.add(i)
            matched_dets.add(j)

        # Unmatched tracks → mark disappeared
        to_remove = []
        for i, tid in enumerate(track_ids):
            if i not in matched_tracks:
                self._tracks[tid].mark_disappeared()
                if self._tracks[tid].disappeared > self._max_disappeared:
                    to_remove.append(tid)
        for tid in to_remove:
            self._deregister(tid)

        # Unmatched detections → register new tracks
        for j in range(n_det):
            if j not in matched_dets:
                self._register(boxes[j], float(confs[j]))

        return list(self._tracks.values())

    def get_smoothed_boxes(self) -> np.ndarray:
        """Return Nx4 array of EMA-smoothed bounding boxes for active tracks."""
        if not self._tracks:
            return np.empty((0, 4), dtype=np.float64)
        return np.array(
            [t.bbox for t in self._tracks.values()],
            dtype=np.float64
        )

    def get_smoothed_centers(self) -> List[List[float]]:
        """Return bottom-center points for all active tracks (for heatmap)."""
        return [list(t.bottom_center) for t in self._tracks.values()]

    def reset(self) -> None:
        """Clear all tracks."""
        self._tracks.clear()
        self._next_id = 0
