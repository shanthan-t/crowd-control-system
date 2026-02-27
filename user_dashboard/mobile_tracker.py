from fastapi import APIRouter
from pydantic import BaseModel
import time

router = APIRouter(prefix="/api/mobile", tags=["Mobile"])

# In-memory storage: user_id -> {lat, lng, last_seen_timestamp}
active_users = {}

class LocationRequest(BaseModel):
    user_id: str
    lat: float
    lng: float
    timestamp: float

@router.post("/location")
def update_location(req: LocationRequest):
    active_users[req.user_id] = {
        "lat": req.lat,
        "lng": req.lng,
        "last_seen": req.timestamp
    }
    return {"status": "ok"}

@router.get("/active")
def get_active_users():
    # Use current time in ms (JS uses ms for Date.now())
    current_time = time.time() * 1000  
    
    # Remove sessions older than 20 seconds
    stale_keys = [k for k, v in active_users.items() if current_time - v["last_seen"] > 20000]
    for k in stale_keys:
        del active_users[k]
        
    positions = [{"lat": u["lat"], "lng": u["lng"]} for u in active_users.values()]
    
    return {
        "active_users": len(active_users),
        "positions": positions
    }
