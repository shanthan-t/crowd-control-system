import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sqlite3 # Kept for legacy if needed, but primary is now MongoDB via shared
import datetime
import socketio
import eventlet
import threading
import serial
import time
import cv2
import base64
from shared.db import get_db # MongoDB Log Access

# --- Configuration ---
DB_NAME = "sentinel_logs.db" # Legacy local backup
SOCKET_PORT = 5000
SERIAL_PORT = "/dev/ttyUSB0" # Default, can be changed via init
SERIAL_BAUD = 9600

# --- Socket.IO Server ---
sio = socketio.Server(cors_allowed_origins='*')
app = socketio.WSGIApp(sio)

class SentinelHub:
    def __init__(self, serial_port=None):
        self.serial_port = serial_port if serial_port else SERIAL_PORT
        self.serial_connection = None
        self.db_lock = threading.Lock()
        
        # 1. Initialize Database
        self.init_db()
        
        # 2. Initialize Serial
        self.init_serial()
        
        # 3. Start Socket Server in background thread
        # Note: Streamlit runs in its own loop. We need to run SocketIO separately or in a thread.
        # For a simple prototype, we can run it in a daemon thread.
        self.socket_thread = threading.Thread(target=self.run_socket_server, daemon=True)
        self.socket_thread.start()

    def init_db(self):
        """Creates the database table if it doesn't exist."""
        with self.db_lock:
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS logs(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    people_count INTEGER,
                    audio_status TEXT,
                    risk_level TEXT
                )
            ''')
            conn.commit()
            conn.close()
            print(f"[Hub] Database {DB_NAME} initialized.")

    def init_serial(self):
        """Attempts to connect to Arduino."""
        try:
            self.serial_connection = serial.Serial(self.serial_port, SERIAL_BAUD, timeout=1)
            time.sleep(2) # Wait for Arduino reset
            print(f"[Hub] Connected to Serial Device at {self.serial_port}")
        except Exception as e:
            print(f"[Hub] Serial connection failed or skipped: {e}")
            self.serial_connection = None

    def run_socket_server(self):
        """Runs the Socket.IO server."""
        try:
            print(f"[Hub] Starting Socket.IO server on port {SOCKET_PORT}...")
            eventlet.wsgi.server(eventlet.listen(('', SOCKET_PORT)), app, log_output=False)
        except Exception as e:
            print(f"[Hub] Socket.IO Server Error: {e}")

    def log_result(self, people_count, audio_status, risk_level):
        """
        Social function to handle all outputs: 
        DB Log -> Terminal -> Socket -> Serial
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 1. Terminal Output
        print(f"[{timestamp}] People: {people_count} | Audio: {audio_status} | Risk: {risk_level}")

        # 2. Database Log (MongoDB + SQLite Backup)
        try:
            # MongoDB (Live for User Dashboard)
            db = get_db()
            db.log_event(people_count, audio_status, risk_level)
            
            # SQLite (Backup)
            with self.db_lock:
                conn = sqlite3.connect(DB_NAME)
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO logs (timestamp, people_count, audio_status, risk_level) VALUES (?, ?, ?, ?)",
                    (timestamp, people_count, audio_status, risk_level)
                )
                conn.commit()
                conn.close()
        except Exception as e:
            print(f"[Hub] DB Error: {e}")

        # 3. Socket Broadcast
        try:
            sio.emit('sentinel_update', {
                'timestamp': timestamp,
                'people_count': people_count,
                'audio_status': audio_status,
                'risk_level': risk_level
            })
        except Exception as e:
            print(f"[Hub] Socket Emit Error: {e}")

        # 4. Serial Alert
        if self.serial_connection:
            try:
                msg = ""
                if risk_level == "LOW": msg = "SAFE\n"
                elif risk_level == "MEDIUM": msg = "WARN\n"
                elif risk_level == "HIGH": msg = "DANGER\n"
                
                if msg:
                    self.serial_connection.write(msg.encode())
            except Exception as e:
                print(f"[Hub] Serial Send Error: {e}")

    def broadcast_frame(self, frame):
        """Encodes and emits a video frame via Socket.IO."""
        try:
            # Resize for performance? Optional.
            # _, buffer = cv2.imencode('.jpg', cv2.resize(frame, (640, 360)), [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            b64_str = base64.b64encode(buffer).decode('utf-8')
            sio.emit('video_frame', b64_str)
        except Exception as e:
            pass # drop frame on error

# Singleton instance placeholder
hub = None

def get_hub():
    global hub
    if hub is None:
        hub = SentinelHub()
    return hub
