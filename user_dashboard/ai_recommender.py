"""
ai_recommender.py — Sentinel AI Tactical Recommendation Engine

Reads live density data from all camera sessions (non-blocking),
computes composite risk scores, analyzes congestion trends, and
generates structured tactical recommendations.

Architecture:
  - Reads session.metrics and session.heatmap_grid atomically
  - No shared locks with detection threads (reads with per-session lock)
  - Runs in API request thread — does not block detection
  - Scalable for multi-camera, multi-zone, future ML integration
"""

import time
import uuid
import threading
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import List, Optional

# ── Configuration ──────────────────────────────────────────────────────────
CROWD_SAFE    = 5
CROWD_DANGER  = 10
TREND_WINDOW  = 30       # samples (~10s at 3Hz)
TREND_RISING  = 0.3      # persons/sec threshold for "RISING"
TREND_FALLING = -0.3     # persons/sec threshold for "FALLING"
IMBALANCE_RATIO = 2.0    # zone imbalance trigger


# ── Data Structures ────────────────────────────────────────────────────────

@dataclass
class Recommendation:
    id: str
    severity: str          # CRITICAL / WARNING / INFO
    title: str
    detail: str
    confidence: float      # 0–1
    camera_id: Optional[str]
    timestamp: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CameraSummary:
    camera_id: str
    label: str
    people: int
    density_pct: float     # 0–100
    running: bool

    def to_dict(self) -> dict:
        return asdict(self)


# ── Trend Tracker ──────────────────────────────────────────────────────────

class _TrendTracker:
    """Rolling window density tracker across all cameras."""

    def __init__(self, window_size: int = TREND_WINDOW):
        self._lock = threading.Lock()
        self._samples: deque = deque(maxlen=window_size)

    def add_sample(self, total_people: int, ts: float):
        with self._lock:
            self._samples.append((ts, total_people))

    def get_slope(self) -> float:
        """Returns persons/second slope. Positive = growing."""
        with self._lock:
            if len(self._samples) < 3:
                return 0.0
            samples = list(self._samples)

        # Simple linear regression slope
        n = len(samples)
        t0 = samples[0][0]
        times = [s[0] - t0 for s in samples]
        values = [s[1] for s in samples]

        dt = times[-1] - times[0]
        if dt < 0.5:  # not enough time span
            return 0.0

        mean_t = sum(times) / n
        mean_v = sum(values) / n
        numerator = sum((t - mean_t) * (v - mean_v) for t, v in zip(times, values))
        denominator = sum((t - mean_t) ** 2 for t in times)

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def get_trend_label(self) -> str:
        slope = self.get_slope()
        if slope > TREND_RISING:
            return "RISING"
        elif slope < TREND_FALLING:
            return "FALLING"
        return "STABLE"


_trend = _TrendTracker()


# ── Risk Scoring ───────────────────────────────────────────────────────────

def _compute_risk_score(total_people: int, max_people_single_cam: int,
                         audio_panic: bool, trend_slope: float) -> float:
    """Composite risk score 0–1. (Legacy fallback for other functions)"""
    # Density ratio (worst single camera)
    density_ratio = min(max_people_single_cam / max(CROWD_DANGER, 1), 1.0)
    # Trend factor (0–0.2)
    trend_factor = min(max(trend_slope / 3.0, 0.0), 0.2)
    # Audio panic factor
    audio_factor = 0.3 if audio_panic else 0.0
    # Weighted composite
    score = (density_ratio * 0.5) + (trend_factor) + (audio_factor)
    return min(max(score, 0.0), 1.0)

def _risk_level(score: float) -> str:
    if score >= 0.7:
        return "CRITICAL"
    if score >= 0.4:
        return "HIGH"
    if score >= 0.2:
        return "MEDIUM"
    return "LOW"

def _calculate_safety_index(current_count: int, trend_slope: float, active_cameras: int) -> dict:
    """
    Computes a mathematical Crowd Safety Index (0-100).
    Weights: 40% Density, 30% Prediction (5min), 30% Growth
    """
    # Guard against 0 active cameras
    cam_count = max(1, active_cameras)
    
    # 1. Base Variables
    # We dynamically approximate capacity. E.g 50 persons per camera view
    capacity_limit = cam_count * max(CROWD_DANGER * 2, 50) 
    
    # 5-minute prediction (slope is persons/sec * 300 seconds)
    predicted_count_5min = max(0, int(current_count + (trend_slope * 300)))
    
    # Growth rate count 
    growth_rate = max(predicted_count_5min - current_count, 0)

    # 2. Score Components
    density_ratio = current_count / capacity_limit
    density_score = min(density_ratio * 100.0, 100.0)
    
    prediction_ratio = predicted_count_5min / capacity_limit
    prediction_score = min(prediction_ratio * 100.0, 100.0)
    
    growth_ratio = growth_rate / capacity_limit
    growth_score = min(growth_ratio * 100.0, 100.0)

    # 3. Final Weighted Score
    final_score = (
        (density_score * 0.4) +
        (prediction_score * 0.3) +
        (growth_score * 0.3)
    )
    
    # Clamp to integer 0-100
    csi = max(0, min(100, int(final_score)))
    
    # Level logic
    if csi <= 30:
        level, color = "SAFE", "green"
    elif csi <= 60:
        level, color = "MODERATE", "yellow"
    elif csi <= 80:
        level, color = "HIGH RISK", "orange"
    else:
        level, color = "CRITICAL", "red"

    return {
        "score": csi,
        "level": level,
        "color": color,
        "metrics": {
            "density_score": round(density_score, 1),
            "prediction_score": round(prediction_score, 1),
            "growth_score": round(growth_score, 1),
            "predicted_count_5min": predicted_count_5min,
            "capacity_limit": capacity_limit
        }
    }


# ── Recommendation Rules ──────────────────────────────────────────────────

def _generate_recommendations(
    camera_summaries: List[CameraSummary],
    total_people: int,
    risk_score: float,
    trend_slope: float,
    trend_label: str,
    audio_panic: bool,
) -> List[Recommendation]:
    """Apply v1 rule-based logic to generate tactical recommendations."""

    recs: List[Recommendation] = []
    now = time.time()

    # Find worst camera
    worst_cam = max(camera_summaries, key=lambda c: c.people, default=None)
    running_cams = [c for c in camera_summaries if c.running]

    # Rule 1: Emergency — high density + audio panic
    if audio_panic and total_people >= CROWD_SAFE:
        recs.append(Recommendation(
            id=f"rec-{uuid.uuid4().hex[:8]}",
            severity="CRITICAL",
            title="Emergency response — potential panic event",
            detail=f"Audio panic detected with {total_people} persons in monitored zones. Initiate emergency protocol immediately.",
            confidence=0.92,
            camera_id=worst_cam.camera_id if worst_cam else None,
            timestamp=now,
        ))

    # Rule 2: Overcrowding — density > 80%
    for cam in running_cams:
        if cam.density_pct >= 80:
            recs.append(Recommendation(
                id=f"rec-{uuid.uuid4().hex[:8]}",
                severity="CRITICAL",
                title="Activate overflow protocol",
                detail=f"{cam.label}: density at {cam.density_pct:.0f}% capacity ({cam.people} persons). Open secondary exits or redirect flow.",
                confidence=0.88,
                camera_id=cam.camera_id,
                timestamp=now,
            ))

    # Rule 3: Rapid influx — trend slope > threshold
    if trend_label == "RISING" and trend_slope > TREND_RISING:
        recs.append(Recommendation(
            id=f"rec-{uuid.uuid4().hex[:8]}",
            severity="WARNING",
            title="Crowd influx detected — pre-position staff",
            detail=f"Density increasing at +{trend_slope:.1f} persons/sec. Deploy additional personnel to high-traffic zones.",
            confidence=0.78,
            camera_id=None,
            timestamp=now,
        ))

    # Rule 4: Zone imbalance — one camera has 2x+ the average
    if len(running_cams) >= 2:
        counts = [c.people for c in running_cams]
        avg = sum(counts) / len(counts) if counts else 1
        for cam in running_cams:
            if avg > 0 and cam.people > avg * IMBALANCE_RATIO and cam.people >= CROWD_SAFE:
                recs.append(Recommendation(
                    id=f"rec-{uuid.uuid4().hex[:8]}",
                    severity="WARNING",
                    title=f"Zone imbalance — redistribute staff",
                    detail=f"{cam.label} has {cam.people} persons ({cam.people / avg:.1f}x the average). Move staff from quieter zones.",
                    confidence=0.72,
                    camera_id=cam.camera_id,
                    timestamp=now,
                ))

    # Rule 5: Moderate congestion
    if not recs and total_people >= CROWD_SAFE:
        recs.append(Recommendation(
            id=f"rec-{uuid.uuid4().hex[:8]}",
            severity="WARNING",
            title="Elevated crowd density",
            detail=f"{total_people} persons detected. Monitor closely and prepare contingency plans.",
            confidence=0.70,
            camera_id=worst_cam.camera_id if worst_cam else None,
            timestamp=now,
        ))

    # Rule 6: Low activity
    if total_people > 0 and total_people < 3 and not recs:
        recs.append(Recommendation(
            id=f"rec-{uuid.uuid4().hex[:8]}",
            severity="INFO",
            title="Low activity — reduce patrol frequency",
            detail=f"Only {total_people} person(s) detected. Consider reallocating staff to other duties.",
            confidence=0.90,
            camera_id=None,
            timestamp=now,
        ))

    # Rule 7: Normal operations (fallback)
    if not recs:
        recs.append(Recommendation(
            id=f"rec-{uuid.uuid4().hex[:8]}",
            severity="INFO",
            title="Normal operations — no action required",
            detail="All zones within safe thresholds. Continue standard monitoring.",
            confidence=0.95,
            camera_id=None,
            timestamp=now,
        ))

    return recs


# ══════════════════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════════════════

def get_recommendations(manager) -> dict:
    """
    Main entry point — called by API endpoint.
    Reads all camera sessions from the MultiCameraManager,
    computes risk + trend + recommendations, returns JSON-ready dict.

    Non-blocking: only acquires per-session locks briefly for atomic reads.
    """
    sessions = manager.list_sessions()
    now = time.time()

    # Build camera summaries
    camera_summaries: List[CameraSummary] = []
    total_people = 0
    max_people_single = 0
    audio_panic = False

    for s in sessions:
        people = s.get("people", 0)
        total_people += people
        max_people_single = max(max_people_single, people)
        density_pct = min((people / max(CROWD_DANGER, 1)) * 100, 100)

        if s.get("audio") == "PANIC":
            audio_panic = True

        camera_summaries.append(CameraSummary(
            camera_id=s["camera_id"],
            label=s.get("label", s.get("source_url", "Unknown")),
            people=people,
            density_pct=round(density_pct, 1),
            running=s.get("running", False),
        ))

    # Update trend tracker
    _trend.add_sample(total_people, now)
    trend_slope = _trend.get_slope()
    trend_label = _trend.get_trend_label()

    # Risk score
    risk_score = _compute_risk_score(
        total_people, max_people_single, audio_panic, trend_slope
    )
    risk_level = _risk_level(risk_score)

    # Crowd Safety Index
    active_cameras = len([c for c in camera_summaries if c.running])
    csi_data = _calculate_safety_index(total_people, trend_slope, active_cameras)

    # Recommendations
    recs = _generate_recommendations(
        camera_summaries, total_people, risk_score,
        trend_slope, trend_label, audio_panic,
    )

    return {
        "risk_score": round(risk_score, 3),
        "risk_level": risk_level,
        "csi": csi_data,
        "trend": trend_label,
        "trend_slope": round(trend_slope, 2),
        "total_people": total_people,
        "recommendations": [r.to_dict() for r in recs],
        "camera_summaries": [c.to_dict() for c in camera_summaries],
        "timestamp": now,
    }
