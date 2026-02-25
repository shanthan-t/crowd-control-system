#!/bin/bash

# Activate Virtual Environment
source venv/bin/activate

echo "=================================================="
echo "   🛡️  SENTINEL CROWD SAFETY SYSTEM LAUNCHER    "
echo "=================================================="

# Function to handle cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping Sentinel System..."
    kill $(jobs -p) 2>/dev/null
    echo "✅ System Stopped."
}

# Trap SIGINT (Ctrl+C) and SIGTERM
trap cleanup SIGINT SIGTERM EXIT

# 2. Start FastAPI Auth Backend
echo "🚀 Starting FastAPI Auth Backend..."
python user_dashboard/api.py &
PID2=$!
echo "   -> Auth API running at:            http://localhost:8000"

sleep 1

# 3. Start React/Tailwind Frontend
echo "🚀 Starting React Frontend..."
cd user_dashboard_web && npm run dev &
PID3=$!
cd ..
echo "   -> React Frontend running at:      http://localhost:5173"

echo "=================================================="
echo "   SYSTEM IS LIVE. PRESS CTRL+C TO STOP.          "
echo "=================================================="

# Wait for all processes
wait $PID2 $PID3
