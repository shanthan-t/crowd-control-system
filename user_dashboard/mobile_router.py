from fastapi import APIRouter
from pydantic import BaseModel
import time

router = APIRouter(prefix="/api/mobile", tags=["Mobile"])

# In-memory storage: session_id -> {lat, lon, last_seen_timestamp}
active_sessions = {}

class PingRequest(BaseModel):
    session_id: str
    lat: float
    lon: float
    timestamp: float

@router.post("/ping")
def ping_mobile(req: PingRequest):
    active_sessions[req.session_id] = {
        "lat": req.lat,
        "lon": req.lon,
        "last_seen_timestamp": req.timestamp
    }
    return {"status": "ok"}

@router.get("/heatmap")
def get_mobile_heatmap():
    current_time = time.time() * 1000  # API receives JS timestamp (ms)
    
    # Remove sessions older than 30 seconds
    stale_keys = [k for k, v in active_sessions.items() if current_time - v["last_seen_timestamp"] > 30000]
    for k in stale_keys:
        del active_sessions[k]
        
    count = len(active_sessions)
    
    # Create simple 50x50 grid
    grid_size = 50
    grid = [[0.0 for _ in range(grid_size)] for _ in range(grid_size)]
    
    if count > 0:
        lats = [s["lat"] for s in active_sessions.values()]
        lons = [s["lon"] for s in active_sessions.values()]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)
        
        # Add slight padding to avoid division by zero
        lat_diff = max(max_lat - min_lat, 0.0001)
        lon_diff = max(max_lon - min_lon, 0.0001)
        
        for s in active_sessions.values():
            # Normalize coordinates to 0-1 range
            nx = (s["lon"] - min_lon) / lon_diff
            ny = (s["lat"] - min_lat) / lat_diff
            
            # Map to grid
            gx = min(int(nx * grid_size), grid_size - 1)
            gy = min(int(ny * grid_size), grid_size - 1)
            
            # Increment nearest grid cell
            grid[gy][gx] += 0.5 
            if grid[gy][gx] > 1.0:
                grid[gy][gx] = 1.0
                
    return {
        "grid": grid,
        "count": count,
        "width": grid_size,
        "height": grid_size,
        "ready": True
    }
