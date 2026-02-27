"""
face_tracker.py — Temporal Tracker for Face Mesh Landmarks
============================================================

Tracks the 468 MediaPipe Face Mesh landmarks across frames.
Maintains temporal stability via Exponential Moving Average (EMA).
Persists lost faces for up to 10 frames to prevent flickering.
"""

import numpy as np
from collections import OrderedDict

# ── Configuration ─────────────────────────────────────────────────────────
MAX_FACE_DISAPPEARED = 10     # frames before a face mask is dropped completely
MATCH_RADIUS         = 100.0  # max centroid distance in pixels to match a face
EMA_ALPHA            = 0.4    # weight of new coords; lower = smoother

class TrackedFace:
    """A tracked face with EMA-smoothed 468 landmarks."""

    __slots__ = ('face_id', 'landmarks', 'centroid', 'disappeared')

    def __init__(self, face_id: int, landmarks: np.ndarray):
        self.face_id = face_id
        # landmarks: shape (N, 2)
        self.landmarks = landmarks.astype(np.float64)
        self.centroid = np.mean(self.landmarks, axis=0)
        self.disappeared = 0

    def update(self, landmarks: np.ndarray):
        """EMA update of landmark coordinates."""
        new_lm = landmarks.astype(np.float64)
        # Verify shape
        if new_lm.shape == self.landmarks.shape:
            self.landmarks = EMA_ALPHA * new_lm + (1.0 - EMA_ALPHA) * self.landmarks
        else:
            self.landmarks = new_lm
            
        self.centroid = np.mean(self.landmarks, axis=0)
        self.disappeared = 0

    def mark_disappeared(self):
        self.disappeared += 1


class FaceMeshTracker:
    """
    Tracking for MediaPipe Face Mesh.
    Similar greedy distance matching to CentroidTracker, but stores and smooths 
    entire facial meshes (list of [x, y] coordinates).
    """

    def __init__(self, max_disappeared: int = MAX_FACE_DISAPPEARED, match_radius: float = MATCH_RADIUS):
        self._next_id = 0
        self._tracks: OrderedDict[int, TrackedFace] = OrderedDict()
        self._max_disappeared = max_disappeared
        self._match_radius = match_radius

    def reset(self):
        self._tracks.clear()
        self._next_id = 0

    def update(self, faces_landmarks: list[np.ndarray]):
        """
        Update tracker with a list of face landmarks.
        faces_landmarks: List of arrays, each of shape (N, 2)
        """
        if not faces_landmarks or len(faces_landmarks) == 0:
            # Mark all as disappeared
            to_remove = []
            for face_id, track in self._tracks.items():
                track.mark_disappeared()
                if track.disappeared > self._max_disappeared:
                    to_remove.append(face_id)
            for face_id in to_remove:
                del self._tracks[face_id]
            return
            
        n_det = len(faces_landmarks)
        
        # If no tracks, register all
        if len(self._tracks) == 0:
            for lms in faces_landmarks:
                t = TrackedFace(self._next_id, lms)
                self._tracks[self._next_id] = t
                self._next_id += 1
            return
            
        # Compute centroids for new detections
        det_centroids = np.zeros((n_det, 2), dtype=np.float64)
        for i, lms in enumerate(faces_landmarks):
            det_centroids[i] = np.mean(lms, axis=0)
            
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
                
        # Greedy matching
        matched_tracks = set()
        matched_dets = set()
        flat_indices = np.argsort(dist, axis=None)
        
        for flat_idx in flat_indices:
            i = flat_idx // n_det
            j = flat_idx % n_det
            
            if i in matched_tracks or j in matched_dets:
                continue
            if dist[i, j] > self._match_radius:
                break
                
            # Match found
            tid = track_ids[i]
            self._tracks[tid].update(faces_landmarks[j])
            matched_tracks.add(i)
            matched_dets.add(j)
            
        # Unmatched tracks -> disappear
        to_remove = []
        for i, tid in enumerate(track_ids):
            if i not in matched_tracks:
                self._tracks[tid].mark_disappeared()
                if self._tracks[tid].disappeared > self._max_disappeared:
                    to_remove.append(tid)
        for tid in to_remove:
            del self._tracks[tid]
            
        # Unmatched dets -> new tracks
        for j in range(n_det):
            if j not in matched_dets:
                t = TrackedFace(self._next_id, faces_landmarks[j])
                self._tracks[self._next_id] = t
                self._next_id += 1

    def get_smoothed_landmarks(self) -> list[np.ndarray]:
        """Returns a list of active mask points."""
        return [t.landmarks for t in self._tracks.values()]

