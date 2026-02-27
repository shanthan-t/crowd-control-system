"""
dispatch.py — Sentinel Automated Staff Dispatch Engine

In-memory dispatch state machine:
  IDLE → PENDING (AI detected) → ACTIVE (admin confirmed) → ASSIGNED (staff accepted) → IDLE

Thread-safe. No external dependencies.
"""

import time
import math
import threading
import uuid
from typing import Optional, Dict, List

# ══════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════════════════

DENSITY_THRESHOLD = 0.8     # Dispatch if count/capacity > this
COOLDOWN_SECONDS = 60       # Min seconds between dispatches
STAFF_TO_DEPLOY = 2         # Recommended staff per dispatch

# Incident zone center (default Room 1 centroid)
INCIDENT_ZONE = {"x": 320, "y": 240}

# ══════════════════════════════════════════════════════════════════════════
#  Staff Registry (demo data)
# ══════════════════════════════════════════════════════════════════════════

_staff_registry: List[Dict] = [
    {"id": "S1", "name": "Rajan Sharma",  "zone": "Zone A — Main Gate",  "x": 120, "y": 340, "status": "available"},
    {"id": "S2", "name": "Priya Nair",    "zone": "Zone B — East Wing",  "x": 480, "y": 150, "status": "available"},
    {"id": "S3", "name": "Vikram Singh",  "zone": "Zone C — West Wing",  "x": 50,  "y": 100, "status": "available"},
    {"id": "S4", "name": "Ananya Gupta",  "zone": "Zone D — Food Court",  "x": 600, "y": 400, "status": "available"},
]

# ══════════════════════════════════════════════════════════════════════════
#  Dispatch State
# ══════════════════════════════════════════════════════════════════════════

_lock = threading.Lock()

_dispatch: Optional[Dict] = None       # Current dispatch object
_last_dispatch_time: float = 0.0        # Timestamp of last dispatch creation


def _calculate_distance(a: Dict, b: Dict) -> float:
    return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2)


def _sorted_staff_by_distance(target: Dict) -> List[Dict]:
    """Return staff list sorted by distance to target, with distance included."""
    result = []
    for s in _staff_registry:
        dist = round(_calculate_distance(s, target), 1)
        result.append({**s, "distance": dist})
    result.sort(key=lambda x: x["distance"])
    return result


# ══════════════════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════════════════

def check_dispatch_required(count: int, capacity: int) -> bool:
    """Hard threshold check — no ML."""
    if capacity <= 0:
        return False
    density_ratio = count / capacity
    return density_ratio > DENSITY_THRESHOLD


def create_dispatch(room: str, count: int) -> Optional[Dict]:
    """
    Create a PENDING dispatch if conditions met and cooldown elapsed.
    Returns the dispatch object or None if suppressed.
    """
    global _dispatch, _last_dispatch_time

    with _lock:
        now = time.time()

        # Cooldown guard
        if now - _last_dispatch_time < COOLDOWN_SECONDS:
            return _dispatch if _dispatch else None

        # Already have an active/pending dispatch
        if _dispatch and _dispatch["status"] in ("pending", "active"):
            return _dispatch

        dispatch_id = f"D-{uuid.uuid4().hex[:6].upper()}"
        staff_sorted = _sorted_staff_by_distance(INCIDENT_ZONE)

        _dispatch = {
            "dispatch_id": dispatch_id,
            "status": "pending",        # pending → active → assigned → completed
            "room": room,
            "count": count,
            "recommended_staff": STAFF_TO_DEPLOY,
            "staff": staff_sorted,
            "assigned_to": None,
            "created_at": now,
            "confirmed_at": None,
            "assigned_at": None,
        }
        _last_dispatch_time = now
        return _dispatch


def confirm_dispatch(dispatch_id: str) -> tuple:
    """Admin confirms dispatch → moves to ACTIVE. Returns (ok, error)."""
    global _dispatch

    with _lock:
        if not _dispatch:
            return False, "No pending dispatch."
        if _dispatch["dispatch_id"] != dispatch_id:
            return False, "Dispatch ID mismatch."
        if _dispatch["status"] != "pending":
            return False, f"Dispatch is already {_dispatch['status']}."

        _dispatch["status"] = "active"
        _dispatch["confirmed_at"] = time.time()
        return True, None


def accept_dispatch(dispatch_id: str, staff_id: str) -> tuple:
    """First staff to accept wins. Returns (ok, error)."""
    global _dispatch

    with _lock:
        if not _dispatch:
            return False, "No active dispatch."
        if _dispatch["dispatch_id"] != dispatch_id:
            return False, "Dispatch ID mismatch."
        if _dispatch["status"] != "active":
            return False, f"Dispatch is {_dispatch['status']}, not active."
        if _dispatch["assigned_to"]:
            return False, "Already assigned to another staff member."

        # Find and validate staff
        staff_member = None
        for s in _staff_registry:
            if s["id"] == staff_id:
                staff_member = s
                break

        if not staff_member:
            return False, "Staff not found."

        # Assign
        _dispatch["status"] = "assigned"
        _dispatch["assigned_to"] = {
            "id": staff_member["id"],
            "name": staff_member["name"],
        }
        _dispatch["assigned_at"] = time.time()

        # Update staff registry status
        staff_member["status"] = "deployed"

        return True, None


def cancel_dispatch(dispatch_id: str) -> tuple:
    """Admin cancels a pending dispatch."""
    global _dispatch

    with _lock:
        if not _dispatch:
            return False, "No dispatch to cancel."
        if _dispatch["dispatch_id"] != dispatch_id:
            return False, "Dispatch ID mismatch."
        if _dispatch["status"] not in ("pending", "active"):
            return False, f"Cannot cancel {_dispatch['status']} dispatch."

        _dispatch["status"] = "cancelled"
        return True, None


def get_dispatch_status() -> Optional[Dict]:
    """Return current dispatch state (thread-safe snapshot)."""
    with _lock:
        if not _dispatch:
            return None
        return {**_dispatch}


def get_staff_list() -> List[Dict]:
    """Return current staff registry."""
    with _lock:
        return [
            {**s, "distance": round(_calculate_distance(s, INCIDENT_ZONE), 1)}
            for s in _staff_registry
        ]


def reset_completed_dispatch():
    """Clear a completed/cancelled dispatch so new ones can be triggered."""
    global _dispatch

    with _lock:
        if _dispatch and _dispatch["status"] in ("assigned", "cancelled", "completed"):
            # Reset assigned staff back to available
            if _dispatch.get("assigned_to"):
                for s in _staff_registry:
                    if s["id"] == _dispatch["assigned_to"]["id"]:
                        s["status"] = "available"
            _dispatch = None
