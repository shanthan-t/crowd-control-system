"""
ai_engine.py — Sentinel AI Tactical Intelligence Engine (v2)

Production-grade recommendation engine:
  Step 1: Density velocity (EMA-smoothed, per-zone rolling history)
  Step 2: Zone criticality weighting (area_type → weight)
  Step 3: Staff-to-crowd ratio logic
  Step 4: 15-second linear forecast
  Step 5: Structured decision output with confidence

Architecture:
  - Non-blocking atomic reads from CameraSession state
  - No external ML libraries
  - Thread-safe
  - Multi-camera / multi-zone support
  - Backward compatible
"""

import time
import uuid
import math
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict


# ══════════════════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════════════════

CROWD_DANGER = 10          # critical threshold (people)
CROWD_SAFE   = 5           # elevated threshold
FORECAST_HORIZON = 15.0    # seconds

ZONE_WEIGHTS: Dict[str, float] = {
    "exit":         1.8,
    "corridor":     1.5,
    "gate":         1.6,
    "closed_room":  1.3,
    "open_ground":  1.0,
}

STAFF_RATIO = 20           # 1 staff per 20 people (ideal)


# ══════════════════════════════════════════════════════════════════════════
#  Data Structures
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class ZoneAssessment:
    """Structured assessment for a single camera/zone."""
    zone_id: str
    label: str
    density: int
    velocity: float
    forecast_density: float
    staff_current: int
    staff_required: int
    risk_score: float
    alert_level: str        # LOW | MEDIUM | HIGH | CRITICAL
    recommendation: str
    confidence: float       # 0–1
    area_type: str
    zone_weight: float
    running: bool

    def to_dict(self) -> dict:
        d = asdict(self)
        d["velocity"] = round(d["velocity"], 3)
        d["forecast_density"] = round(d["forecast_density"], 1)
        d["risk_score"] = round(d["risk_score"], 3)
        d["confidence"] = round(d["confidence"], 3)
        d["zone_weight"] = round(d["zone_weight"], 2)
        return d


# ══════════════════════════════════════════════════════════════════════════
#  Step 1 — Density Velocity
# ══════════════════════════════════════════════════════════════════════════

def _read_velocity(session_dict: dict) -> float:
    """Read pre-computed EMA velocity from session state."""
    return session_dict.get("velocity", 0.0)


# ══════════════════════════════════════════════════════════════════════════
#  Step 2 — Zone Criticality Weighting
# ══════════════════════════════════════════════════════════════════════════

def _get_zone_weight(area_type: str) -> float:
    """Look up zone weight from area_type. Defaults to 1.0."""
    key = area_type.lower().replace(" ", "_") if area_type else "open_ground"
    return ZONE_WEIGHTS.get(key, 1.0)


def _compute_risk_score(density_norm: float, velocity_norm: float,
                         zone_weight_norm: float) -> float:
    """
    Weighted risk score:
      risk = density_norm * 0.4 + velocity_norm * 0.4 + zone_weight_norm * 0.2
    All inputs should be normalized to 0–1.
    """
    score = (density_norm * 0.4) + (velocity_norm * 0.4) + (zone_weight_norm * 0.2)
    return max(0.0, min(1.0, score))


def compute_csi(current_count, capacity_limit):
    try:
        capacity = capacity_limit if capacity_limit and capacity_limit > 0 else 100
        count = current_count if current_count is not None else 0

        density_ratio = min(count / capacity, 1.0)
        csi = int(density_ratio * 100)

        return max(0, min(csi, 100))

    except Exception:
        return 0


# ══════════════════════════════════════════════════════════════════════════
#  Step 3 — Staff Ratio Logic
# ══════════════════════════════════════════════════════════════════════════

def _ideal_staff(density: int) -> int:
    """Ideal staff count: ceil(density / STAFF_RATIO)."""
    if density <= 0:
        return 0
    return math.ceil(density / STAFF_RATIO)


def _staff_recommendation(current_staff: int, ideal: int) -> str:
    """Generate staff recommendation string."""
    if ideal == 0:
        return "Staff Adequate"
    if current_staff < ideal:
        deficit = ideal - current_staff
        return f"Deploy Additional Staff (+{deficit})"
    if current_staff > ideal + 1:
        surplus = current_staff - ideal
        return f"Rebalance Staff (-{surplus})"
    return "Staff Adequate"


# ══════════════════════════════════════════════════════════════════════════
#  Step 4 — 15-Second Forecast
# ══════════════════════════════════════════════════════════════════════════

def _forecast_density(current: int, velocity: float,
                       horizon: float = FORECAST_HORIZON) -> float:
    """Linear projection: forecast = current + velocity × horizon. Clamped ≥ 0."""
    return max(0.0, current + velocity * horizon)


# ══════════════════════════════════════════════════════════════════════════
#  Step 5 — Decision Logic
# ══════════════════════════════════════════════════════════════════════════

def _alert_level(risk_score: float) -> str:
    if risk_score >= 0.75:
        return "CRITICAL"
    if risk_score >= 0.50:
        return "HIGH"
    if risk_score >= 0.25:
        return "MEDIUM"
    return "LOW"


def _build_recommendation(
    density: int,
    velocity: float,
    forecast: float,
    staff_current: int,
    staff_required: int,
    alert: str,
    area_type: str,
    label: str,
) -> str:
    """Generate primary recommendation string based on all factors."""

    parts: List[str] = []

    # Preemptive forecast warning
    if forecast >= CROWD_DANGER and density < CROWD_DANGER:
        parts.append(
            f"Preemptive: forecast reaches {forecast:.0f} in {FORECAST_HORIZON:.0f}s — "
            f"prepare overflow protocol"
        )

    # Current overcrowding
    if density >= CROWD_DANGER:
        parts.append(
            f"Critical density ({density} persons) — activate overflow protocol"
        )
    elif density >= CROWD_SAFE:
        parts.append(f"Elevated density ({density} persons) — monitor closely")

    # Rapid influx
    if velocity > 0.5:
        parts.append(f"Rapid influx +{velocity:.1f}/s — pre-position staff")
    elif velocity < -0.5:
        parts.append(f"Crowd dispersing {velocity:.1f}/s — maintain exits clear")

    # Staff logic
    staff_rec = _staff_recommendation(staff_current, staff_required)
    if staff_rec != "Staff Adequate":
        parts.append(staff_rec)

    # Zone-specific
    if area_type in ("exit", "gate") and density >= CROWD_SAFE:
        parts.append(f"High-criticality zone ({area_type}) — prioritize flow")

    if not parts:
        parts.append("Normal operations — no action required")

    return "; ".join(parts)


def _confidence(density_norm: float, velocity_norm: float,
                zone_weight_norm: float) -> float:
    """
    confidence = min(1.0,
        0.4 * density_norm + 0.4 * velocity_norm + 0.2 * zone_weight_norm
    )
    """
    return min(1.0, 0.4 * density_norm + 0.4 * velocity_norm + 0.2 * zone_weight_norm)


# ══════════════════════════════════════════════════════════════════════════
#  Assessment Builder — per zone
# ══════════════════════════════════════════════════════════════════════════

def _assess_zone(session_dict: dict) -> ZoneAssessment:
    """Build a full ZoneAssessment from a session summary dict."""

    zone_id   = session_dict["camera_id"]
    label     = session_dict.get("label", zone_id)
    density   = session_dict.get("people", 0)
    velocity  = _read_velocity(session_dict)
    area_type = session_dict.get("area_type", "open_ground")
    staff_cur = session_dict.get("staff_count", 0)
    running   = session_dict.get("running", False)

    # Normalizations (0–1)
    density_norm  = min(density / max(CROWD_DANGER, 1), 1.0)
    velocity_norm = min(abs(velocity) / 3.0, 1.0)   # 3 persons/s = max
    zone_weight   = _get_zone_weight(area_type)
    zone_w_norm   = min((zone_weight - 1.0) / 0.8, 1.0)  # 1.0→0, 1.8→1.0

    # Risk score
    risk_score = _compute_risk_score(density_norm, velocity_norm, zone_w_norm)

    # Alert level
    alert = _alert_level(risk_score)

    # Staff
    staff_required = _ideal_staff(density)

    # Forecast
    forecast = _forecast_density(density, velocity)

    # Recommendation
    rec = _build_recommendation(
        density, velocity, forecast,
        staff_cur, staff_required, alert,
        area_type, label,
    )

    # Confidence
    conf = _confidence(density_norm, velocity_norm, zone_w_norm)

    return ZoneAssessment(
        zone_id=zone_id,
        label=label,
        density=density,
        velocity=velocity,
        forecast_density=forecast,
        staff_current=staff_cur,
        staff_required=staff_required,
        risk_score=risk_score,
        alert_level=alert,
        recommendation=rec,
        confidence=conf,
        area_type=area_type,
        zone_weight=zone_weight,
        running=running,
    )


# ══════════════════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════════════════

def get_recommendations(manager) -> dict:
    """
    Main entry point — called by API endpoint.

    Reads all camera sessions from MultiCameraManager,
    runs per-zone assessment, aggregates into global risk.

    Non-blocking: only acquires per-session locks briefly via list_sessions().
    """
    sessions = manager.list_sessions()
    now = time.time()

    if not sessions:
        return {
            "risk_score": 0.0,
            "risk_level": "LOW",
            "trend": "STABLE",
            "trend_slope": 0.0,
            "total_people": 0,
            "forecast_total": 0.0,
            "staff_total": 0,
            "staff_required_total": 0,
            "recommendations": [],
            "zone_assessments": [],
            "timestamp": now,
        }

    # ── Per-zone assessments ──────────────────────────────────────────
    assessments: List[ZoneAssessment] = []
    for s in sessions:
        za = _assess_zone(s)
        assessments.append(za)

    # ── Global aggregation ─────────────────────────────────────────────
    total_people = sum(z.density for z in assessments)
    total_forecast = sum(z.forecast_density for z in assessments)
    total_staff = sum(z.staff_current for z in assessments)
    total_staff_req = sum(z.staff_required for z in assessments)

    # Global risk = worst-case zone risk
    global_risk = max(z.risk_score for z in assessments) if assessments else 0.0
    global_alert = _alert_level(global_risk)

    # Global velocity (weighted average)
    running_zones = [z for z in assessments if z.running]
    if running_zones:
        avg_velocity = sum(z.velocity for z in running_zones) / len(running_zones)
    else:
        avg_velocity = 0.0

    # Trend label
    if avg_velocity > 0.3:
        trend = "RISING"
    elif avg_velocity < -0.3:
        trend = "FALLING"
    else:
        trend = "STABLE"

    # ── Build top-level recommendations from assessments ───────────────
    recs = []
    for za in assessments:
        if za.alert_level in ("CRITICAL", "HIGH") or za.recommendation != "Normal operations — no action required":
            severity = "CRITICAL" if za.alert_level == "CRITICAL" else \
                       "WARNING" if za.alert_level in ("HIGH", "MEDIUM") else "INFO"
            recs.append({
                "id": f"rec-{uuid.uuid4().hex[:8]}",
                "severity": severity,
                "title": f"{za.label}: {za.alert_level}",
                "detail": za.recommendation,
                "confidence": za.confidence,
                "camera_id": za.zone_id,
                "timestamp": now,
                "forecast_density": round(za.forecast_density, 1),
                "staff_action": _staff_recommendation(za.staff_current, za.staff_required),
            })

    # Fallback: normal ops
    if not recs:
        recs.append({
            "id": f"rec-{uuid.uuid4().hex[:8]}",
            "severity": "INFO",
            "title": "Normal operations",
            "detail": "All zones within safe thresholds. Continue standard monitoring.",
            "confidence": 0.95,
            "camera_id": None,
            "timestamp": now,
            "forecast_density": round(total_forecast, 1),
            "staff_action": "Staff Adequate",
        })

    # Safety Index Score
    cam_count = max(1, len(running_zones))
    capacity_limit = cam_count * max(CROWD_DANGER * 2, 50)
    csi_value = compute_csi(total_people, capacity_limit)
    csi_data = {
        "current_count": total_people or 0,
        "capacity_limit": capacity_limit or 100,
        "crowd_safety_index": csi_value
    }

    # ── Dispatch trigger check ─────────────────────────────────────────
    dispatch_alert = None
    try:
        from user_dashboard import dispatch
        if dispatch.check_dispatch_required(csi_value, total_people, capacity_limit):
            pending = dispatch.create_dispatch("Room 1", csi_value, total_people)
            if pending and pending["status"] in ("pending", "active", "assigned"):
                dispatch_alert = pending
        else:
            # Clear completed dispatches when conditions normalize
            dispatch.reset_completed_dispatch()
    except Exception:
        pass  # dispatch module import failure should never break recommendations

    return {
        "risk_score": round(global_risk, 3),
        "risk_level": global_alert,
        "csi": csi_data,
        "trend": trend,
        "trend_slope": round(avg_velocity, 2),
        "total_people": total_people,
        "forecast_total": round(total_forecast, 1),
        "staff_total": total_staff,
        "staff_required_total": total_staff_req,
        "recommendations": recs,
        "zone_assessments": [z.to_dict() for z in assessments],
        "dispatch_alert": dispatch_alert,
        "timestamp": now,
    }

