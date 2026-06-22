import cv2
import pickle
import numpy as np
from flask import Flask, render_template, Response, request, redirect, url_for, send_from_directory, send_file, jsonify, session
from insightface.app import FaceAnalysis
import os
import sqlite3
from datetime import datetime, timedelta
import time
import serial
import threading
import json
import pandas as pd
from collections import deque
from collections import defaultdict
import calendar
from datetime import datetime
import mediapipe as mp
from flask_socketio import SocketIO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import face_landmarker
import numpy as np
import cv2
from mediapipe.tasks.python.vision import FaceLandmarkerOptions, RunningMode
from ultralytics import YOLO
import csv
import io
from flask import Response

# MediaPipe Tasks API imports
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
RunningMode = vision.RunningMode

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")  # enable CORS if needed
app.secret_key = "eduvision_secret"

DB_LOGIN = "eduvision_users.db"

stats = defaultdict(int)

# Global stats dictionary
stats = {
    "noise_level": 0,
    "yawning_count": 0,
    "phone_count": 0,
    "violation_count": 0,
    "unknown_person": 0
}

CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.4
HAND_PHONE_MAX_DIST = 80

# =========================
# ARDUINO SERIAL CONFIG
# =========================
SERIAL_PORT = "COM3"
BAUD_RATE = 9600

arduino_id = None
serial_lock = threading.Lock()

# =========================
# FRAME BUFFERS PER CAMERA
# =========================
frame_buffers = {}
frame_locks = {}

# Cooldown settings
DETECTION_COOLDOWN = 15  # seconds (adjust as needed)

# Store last detection time
last_detection_time = {}  
# key format: (cam_id, student_id)

UNKNOWN_ALERT_COOLDOWN = 60  # seconds
last_unknown_alert = {}

# =========================
# CURRENT SELECTED subjects
# =========================
current_subjects = None

fps_counter = 0
fps = 0
fps_time = time.time()

# ------------------- CONFIG -------------------
MODEL_PATH = "C:\\Users\\kirstin\\Downloads\\face_landmarker.task"  # change if needed
YAWN_THRESHOLD = 0.30
EYE_CLOSED_THRESHOLD = 0.27

# Face landmark indices
UPPER_LIP = 13

LOWER_LIP = 14
LEFT_MOUTH = 78
RIGHT_MOUTH = 308
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ----------------- INIT MODEL -----------------
latest_landmarks = {}
landmarkers = {}
def create_landmarker(cam_id):

    def callback(result, output_image, timestamp_ms):
        latest_landmarks[cam_id] = result

    base_options = BaseOptions(model_asset_path=MODEL_PATH)

    options = FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.LIVE_STREAM,
        result_callback=callback,   # REQUIRED
        num_faces=10
    )

    return FaceLandmarker.create_from_options(options)

# ----------------- HELPER FUNCTIONS -----------------
def mouth_open_ratio(landmarks, w, h):
    upper = landmarks[UPPER_LIP]
    lower = landmarks[LOWER_LIP]
    left = landmarks[LEFT_MOUTH]
    right = landmarks[RIGHT_MOUTH]
    vertical = np.linalg.norm([(upper.x - lower.x) * w, (upper.y - lower.y) * h])
    horizontal = np.linalg.norm([(left.x - right.x) * w, (left.y - right.y) * h])
    return vertical / horizontal

def eye_aspect_ratio(landmarks, eye_idx, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_idx]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2.0 * C)

@app.route("/set_subjects", methods=["POST"])
def set_subjects():
    global current_subjects

    data = request.json
    subject = data.get("subjects")

    # Normalize everything to "None"
    if not subject or subject.strip() == "" or subject == "None":
        current_subjects = "None"
    else:
        current_subjects = subject.strip()

    print("[INFO] Current subjects set to:", current_subjects)

    return jsonify({"status": "success"})

def send_sms_alert():
    try:
        with serial_lock:
            if arduino_ser is not None and arduino_ser.is_open:
                arduino_ser.write(b"ALERT_UNKNOWN\n")
                print("[SMS] Alert command sent to Arduino")
            else:
                print("[SMS ERROR] Arduino serial not ready")
    except Exception as e:
        print("[SMS ERROR]", e)

def camera_reader(cam_id, source):
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep only the latest frame
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
    cap.set(cv2.CAP_PROP_FPS, 10)

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        with frame_locks[cam_id]:
            frame_buffers[cam_id].append(frame)

import cv2
import time

def rtsp_reader(cam_id, source):
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)

    # ---- OPTIMIZATION SETTINGS ----
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # reduce latency
    cap.set(cv2.CAP_PROP_FPS, 10)         # optional: control FPS
    reconnect_delay = 1                   # seconds

    while True:
        if not cap.isOpened():
            cap.release()
            time.sleep(reconnect_delay)
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            continue

        ret, frame = cap.read()

        if not ret:
            # ---- HANDLE FRAME DROP / RECONNECT ----
            cap.release()
            time.sleep(reconnect_delay)
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            continue

        # ---- OPTIONAL: FRAME RESIZE (reduce CPU load) ----
        frame = cv2.resize(frame, (1280, 720))

        # ---- THREAD-SAFE BUFFER WRITE ----
        with frame_locks[cam_id]:
            frame_buffers[cam_id] = [frame]

        # ---- SMALL SLEEP TO PREVENT CPU OVERLOAD ----
        time.sleep(0.001)

def read_arduino():
    global arduino_id, arduino_ser

    try:
        arduino_ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print("[INFO] Arduino connected")
    except Exception as e:
        print(f"[ERROR] Arduino not found: {e}")
        return

    while True:
        try:
            if arduino_ser.in_waiting > 0:
                raw = arduino_ser.readline().decode(errors='ignore').strip()
                if raw:
                    with serial_lock:
                        arduino_id = raw
                    print(f"[RFID READ] Arduino ID updated: {arduino_id}")
            time.sleep(0.1)
        except Exception as e:
            print(f"[ERROR] Serial read failed: {e}")
            time.sleep(1)

DB_PATH = "face_db.pkl"
LOG_DB = "face_logs.db"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT_DIR = os.path.join(BASE_DIR, "face_snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# =========================
# LOAD FACE DATABASE
# =========================
if os.path.exists(DB_PATH):
    with open(DB_PATH, "rb") as f:
        face_db = pickle.load(f)
else:
    face_db = {}

# =========================
# INIT LOG DATABASE
# =========================
def init_logs_db():
    conn = sqlite3.connect(LOG_DB)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            name TEXT,
            course TEXT,
            subjects TEXT,
            time_in TEXT,
            score REAL,
            image TEXT
        )
    """)
    # Attendance table
    c.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            name TEXT,
            course TEXT,
            subjects TEXT,
            time_in TEXT,
            score REAL,
            status TEXT,
            image TEXT
        )
    """)
    conn.commit()
    conn.close()

init_logs_db()

def init_visitor_db():
    conn = sqlite3.connect(LOG_DB)
    c = conn.cursor()

    # Drop old table if exists
    #c.execute("DROP TABLE IF EXISTS visitors")

    # Create new visitors table with proper columns
    c.execute("""
        CREATE TABLE IF NOT EXISTS visitors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            visitor_id TEXT UNIQUE,
            name TEXT,
            visitor_type TEXT,
            purpose TEXT,
            expiration_date TEXT,
            embedding BLOB
        )
    """)
    conn.commit()
    conn.close()

init_visitor_db()

def init_teachers_table():
    conn = sqlite3.connect(LOG_DB)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS teachers (
        teacher_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        department TEXT,
        subjects TEXT,
        time_in TEXT,
        time_out TEXT,
        embedding BLOB
    )
    """)

    conn.commit()
    conn.close()
    print("✅ Teachers table ready (no deletion)")

# Call this when app starts
init_teachers_table()

def init_subjects_db():
    conn = sqlite3.connect(LOG_DB)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS subjects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subjects_name TEXT UNIQUE,
            schedule TEXT,
            teacher_name TEXT,
            room_number TEXT,
            rows INTEGER,
            columns INTEGER
        )
    """)

    conn.commit()
    conn.close()

init_subjects_db()

# Create behavior_logs table if it doesn't exist
def init_behavior_logs_table():
    conn = sqlite3.connect(LOG_DB)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS behavior_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            yawning_count INTEGER DEFAULT 0,
            phone_count INTEGER DEFAULT 0,
            violation_count INTEGER DEFAULT 0,
            unknown_person_count INTEGER DEFAULT 0,
            noise_level REAL DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()

# Call this once at app startup
init_behavior_logs_table()

# -----------------------------
# DATABASE INIT
# -----------------------------
def init_db():
    conn = sqlite3.connect(DB_LOGIN)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT,
        role TEXT
    )
    """)

    # default admin
    c.execute("SELECT * FROM users WHERE username='admin'")
    if not c.fetchone():
        c.execute(
            "INSERT INTO users(username,password,role) VALUES(?,?,?)",
            ("admin","admin123","admin")
        )

    # default teacher
    c.execute("SELECT * FROM users WHERE username='teacher'")
    if not c.fetchone():
        c.execute(
            "INSERT INTO users(username,password,role) VALUES(?,?,?)",
            ("teacher","teacher123","teacher")
        )

    conn.commit()
    conn.close()

init_db()

def auto_delete_expired_visitors():
    conn = sqlite3.connect(LOG_DB)
    c = conn.cursor()

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Get expired visitors (for debug/logging)
    c.execute(
        "SELECT visitor_id FROM visitors WHERE expiration_date <= ?",
        (now,)
    )
    expired = c.fetchall()

    if expired:
        print("[AUTO-DELETE] Expired visitors:", [v[0] for v in expired])

    # Delete expired visitors
    c.execute(
        "DELETE FROM visitors WHERE expiration_date <= ?",
        (now,)
    )

    conn.commit()
    conn.close()

def calculate_expiration(visitor_type):
    now = datetime.now()
    if visitor_type == "Temporary":
        # Expires today at 23:59:59
        return now.replace(hour=23, minute=59, second=59, microsecond=0)
    else:
        # ~1 semester (4 months) safely
        month = now.month + 4
        year = now.year
        if month > 12:
            month -= 12
            year += 1
        day = min(now.day, 28)  # prevent invalid day
        return datetime(year, month, day, now.hour, now.minute, now.second)

# =========================
# INSIGHTFACE
# =========================
face_app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

def detect_camera(max_index=3):
    import os
    import cv2

    # ✅ GLOBAL FFmpeg LOW-LATENCY SETTINGS
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0|analyzeduration;0|probesize;32"
    )

    for i in range(max_index):

        cap = cv2.VideoCapture(i, cv2.CAP_FFMPEG)

        if cap.isOpened():

            # ✅ REDUCE BUFFER (CRITICAL FOR LAG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # ✅ LOWER RESOLUTION (FASTER PROCESSING)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            # ✅ LIMIT FPS (STABLE + LESS CPU)
            cap.set(cv2.CAP_PROP_FPS, 10)

            # ✅ FAST FRAME GRAB (DISCARD OLD FRAMES)
            for _ in range(3):
                cap.grab()

            return cap

    # fallback
    cap = cv2.VideoCapture(0, cv2.CAP_FFMPEG)

    # apply same optimizations
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 10)

    for _ in range(3):
        cap.grab()

    return cap

TAPO_RTSP3 = "rtsp://tapoc260:gerald123@192.168.1.57:554/stream2"

CAMERA_SOURCES = {
    0: TAPO_RTSP3
}

for cam_id in CAMERA_SOURCES.keys():
    frame_buffers[cam_id] = deque(maxlen=5)
    frame_locks[cam_id] = threading.Lock()

@app.route("/detection_stream")
def detection_stream():
    def event_stream():
        while True:
            for cam_id, buffer in frame_buffers.items():
                with frame_locks[cam_id]:
                    if not buffer:
                        continue
                    frame = buffer[-1].copy()

                faces = face_app.get(frame)
                for face in faces:
                    embedding = face.embedding
                    student_id = "Unknown"
                    name = "Unknown"
                    role_label = "Unknown"
                    max_score = 0

                    # Match with known faces
                    for sid, info in face_db.items():
                        score = cosine_similarity(embedding, info["embedding"])
                        if score > max_score and score > 0.55:
                            max_score = score
                            student_id = info["student_id"]
                            name = info["name"]
                            role_label = "Student"

                    # Match with visitors if unknown
                    if student_id == "Unknown":
                        conn = sqlite3.connect(LOG_DB)
                        c = conn.cursor()
                        c.execute("SELECT visitor_id, name, visitor_type, expiration_date, embedding FROM visitors")
                        for row in c.fetchall():
                            v_id, v_name, v_type, exp, v_embedding = row
                            v_embedding = pickle.loads(v_embedding)
                            score = cosine_similarity(embedding, v_embedding)
                            if score > max_score and score > 0.55:
                                if datetime.now() <= datetime.strptime(exp, "%Y-%m-%d %H:%M:%S"):
                                    max_score = score
                                    student_id = v_id
                                    name = v_name
                                    role_label = f"Visitor ({v_type})"
                        conn.close()

                    # Save snapshot
                    x1, y1, x2, y2 = map(int, face.bbox)
                    face_crop = frame[y1:y2, x1:x2]
                    snapshot_path = None
                    if face_crop.size != 0:
                        ts = time.strftime("%Y%m%d_%H%M%S")
                        snapshot_path = f"{student_id}_{ts}.jpg"
                        cv2.imwrite(os.path.join(SNAPSHOT_DIR, snapshot_path), face_crop)

                    # ---- Cooldown check to prevent popup spam ----
                    now = time.time()
                    key = (cam_id, student_id)

                    last_time = last_detection_time.get(key, 0)

                    if now - last_time >= DETECTION_COOLDOWN:
                        last_detection_time[key] = now

                        data = {
                            "cam_id": cam_id,
                            "name": name,
                            "role": role_label,
                            "snapshot": f"/face_snapshots/{snapshot_path}" if snapshot_path else ""
                        }

                        yield f"data: {json.dumps(data)}\n\n"

                    # 🚨 UNKNOWN FACE ALERT
                    if student_id == "Unknown":
                        alert_key = cam_id
                        last_alert = last_unknown_alert.get(alert_key, 0)

                        if now - last_alert >= UNKNOWN_ALERT_COOLDOWN:
                            last_unknown_alert[alert_key] = now

                            snapshot_full_path = (
                                os.path.join(SNAPSHOT_DIR, snapshot_path)
                                if snapshot_path else None
                            )
                            send_sms_alert()

            time.sleep(1)  # small delay to prevent flooding

    return Response(event_stream(), mimetype="text/event-stream")

# =========================
# FACE TRACKING SYSTEM (FIX SWITCHING)
# =========================
tracked_faces = {}  # track_id -> {bbox, last_seen, embedding}
next_track_id = 0

TRACK_DISTANCE_THRESHOLD = 80   # pixels
TRACK_TIMEOUT = 2.0             # seconds


def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def assign_track_id(bbox, embedding):
    global next_track_id

    cx, cy = get_center(bbox)
    current_time = time.time()

    best_id = None
    best_dist = float("inf")

    # Find closest existing track
    for tid, data in tracked_faces.items():
        tx, ty = get_center(data["bbox"])
        dist = np.sqrt((cx - tx)**2 + (cy - ty)**2)

        if dist < best_dist and dist < TRACK_DISTANCE_THRESHOLD:
            best_dist = dist
            best_id = tid

    # Assign existing track
    if best_id is not None:
        tracked_faces[best_id]["bbox"] = bbox
        tracked_faces[best_id]["last_seen"] = current_time
        tracked_faces[best_id]["embedding"] = embedding
        return best_id

    # Create new track
    track_id = next_track_id
    tracked_faces[track_id] = {
        "bbox": bbox,
        "last_seen": current_time,
        "embedding": embedding
    }
    next_track_id += 1

    return track_id


def cleanup_tracks():
    current_time = time.time()
    to_delete = []

    for tid, data in tracked_faces.items():
        if current_time - data["last_seen"] > TRACK_TIMEOUT:
            to_delete.append(tid)

    for tid in to_delete:
        tracked_faces.pop(tid, None)

# =========================
# FACE STABILITY SYSTEM
# =========================
from collections import Counter

# average embeddings across frames
embedding_memory = defaultdict(lambda: deque(maxlen=3))

# stabilize identity across frames
recognition_memory = defaultdict(lambda: deque(maxlen=3))

# Load YOLO models for gesture & phone detection
phone_model = YOLO("yolo26l.pt")
phone_model.to("cuda")
pose_model = YOLO("yolo26n-pose.pt")
pose_model.to("cuda")

def generate_frames(cam_id):

    global fps_counter, fps, fps_time

    frame_count = 0

    # --------------------------
    # Initialize last_timestamps for MediaPipe
    # --------------------------
    global last_timestamps
    if 'last_timestamps' not in globals():
        last_timestamps = {}
    if last_timestamps.get(cam_id) is None:
        last_timestamps[cam_id] = 0

    auto_delete_expired_visitors()

    source = CAMERA_SOURCES.get(cam_id)
    if source is None:
        print(f"[ERROR] Camera ID {cam_id} not configured")
        return

    # Initialize MediaPipe LIVE_STREAM landmarker per camera
    if cam_id not in landmarkers:
        landmarkers[cam_id] = create_landmarker(cam_id)
        print(f"[INFO] MediaPipe LIVE_STREAM initialized for cam {cam_id}")

    # Enable WAL mode once
    conn_init = sqlite3.connect(LOG_DB, timeout=10, check_same_thread=False)
    conn_init.execute("PRAGMA journal_mode=WAL;")
    conn_init.execute("PRAGMA synchronous=NORMAL;")
    conn_init.close()

    # --------------------------
    # LOAD VISITORS ONCE ONLY
    # --------------------------
    visitors_db = []
    try:
        conn_v = sqlite3.connect(LOG_DB, timeout=10, check_same_thread=False)
        c_v = conn_v.cursor()
        c_v.execute("""
            SELECT visitor_id, name, visitor_type, expiration_date, embedding
            FROM visitors
        """)
        rows = c_v.fetchall()
        for row in rows:
            v_id, v_name, v_type, exp, v_embedding = row
            visitors_db.append({
                "visitor_id": v_id,
                "name": v_name,
                "type": v_type,
                "expiration": exp,
                "embedding": pickle.loads(v_embedding)
            })
        conn_v.close()
        print(f"[INFO] Loaded {len(visitors_db)} visitors")
    except Exception as e:
        print("[ERROR loading visitors]", e)

    # --------------------------
    # LOAD TEACHERS ONCE ONLY
    # --------------------------
    teachers_db = []
    try:
        conn_t = sqlite3.connect(LOG_DB, timeout=10, check_same_thread=False)
        c_t = conn_t.cursor()
        c_t.execute("""
            SELECT teacher_id, name, department, subjects, embedding
            FROM teachers
        """)
        rows = c_t.fetchall()
        for row in rows:
            t_id, t_name, t_department, t_subjects, t_embedding = row
            teachers_db.append({
                "teacher_id": t_id,
                "name": t_name,
                "department": t_department,
                "subjects": t_subjects.split(",") if t_subjects else [],
                "embedding": pickle.loads(t_embedding)
            })
        conn_t.close()
        print(f"[INFO] Loaded {len(teachers_db)} teachers")
    except Exception as e:
        print("[ERROR loading teachers]", e)

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"

    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep only the latest frame
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 10)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {cam_id}")
        return

    # --------------------------
    # Yawning detection buffers
    # --------------------------
    yawning_buffer = {}
    yawn_counts = {}

    MAR_THRESHOLD = 0.25
    MAR_FRAMES = 3

    BONES = [
        (0,1),(0,2),       # nose to eyes
        (1,3),(2,4),       # eyes to ears
        (5,7),(7,9),       # left arm
        (6,8),(8,10),      # right arm
        (5,6),(5,11),(6,12), # torso
        (11,13),(13,15),    # left leg
        (12,14),(14,16)     # right leg
    ]

    mp_result = None

    while True:

        fps_counter += 1
        if (time.time() - fps_time) >= 1.0:
            fps = fps_counter
            fps_counter = 0
            fps_time = time.time()

        landmarks = None
        frame_count += 1

        if mp_result is not None:
            for lm_set in mp_result.face_landmarks:
                # Get bounding box of landmarks
                xs = [lm.x * w for lm in lm_set]
                ys = [lm.y * h for lm in lm_set]

                min_x, max_x = int(min(xs)), int(max(xs))
                min_y, max_y = int(min(ys)), int(max(ys))

                # Check overlap with InsightFace bbox
                if (x1 < max_x and x2 > min_x and y1 < max_y and y2 > min_y):
                    landmarks = lm_set
                    break

        stats["unknown_person"] = 0
        stats["yawning_count"] = 0
        stats["phone_count"] = 0
        stats["violation_count"] = 0

        subjects_selected = current_subjects

        with frame_locks[cam_id]:
            if not frame_buffers[cam_id]:
                socketio.sleep(0.01)  # or time.sleep(0.01)
                continue
            frame = frame_buffers[cam_id][-1].copy()

        h, w, _ = frame.shape

        if frame_count % 3 == 0:
            # --- YOLO: Phone detection ---
            phone_results = phone_model.predict(frame, conf=0.5, iou=0.4, verbose=False, profile=False)
            phone_boxes = []
            for result in phone_results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    label = phone_model.names[cls_id]
                    conf = float(box.conf[0])
                    if label == "cell phone" and conf >= 0.5:
                        stats['phone_count'] += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        phone_boxes.append(((x1,y1,x2,y2),conf))
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                        cv2.putText(frame,f"{label} {conf:.2f}",(x1,y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

            # --- YOLO: Pose detection ---
            pose_results = pose_model.predict(frame, conf=0.5, iou=0.4, verbose=False, profile=False)
        else:
            phone_results = []
            pose_results = []

        faces = face_app.get(frame)
        cleanup_tracks()
        conn = sqlite3.connect(LOG_DB, timeout=10, check_same_thread=False)
        c = conn.cursor()

        # --------------------------
        # MediaPipe FULL FRAME (better)
        # --------------------------
        rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_full)

        ts_now = int(time.time() * 1000)
        timestamp = max(ts_now, last_timestamps[cam_id] + 1)
        last_timestamps[cam_id] = timestamp

        landmarkers[cam_id].detect_async(mp_image, timestamp)

        mp_result = latest_landmarks.get(cam_id)

        for face in faces:

            x1, y1, x2, y2 = map(int, face.bbox)
            # --------------------------
            # EMBEDDING STABILIZATION
            # --------------------------
            embedding = face.embedding
            embedding = embedding / np.linalg.norm(embedding)

            track_id = assign_track_id((x1, y1, x2, y2), embedding)

            # store embeddings for averaging
            embedding_memory[(cam_id, track_id)].append(embedding)

            # average embedding (airport technique)
            embedding = np.mean(embedding_memory[(cam_id, track_id)], axis=0)
            embedding = embedding / np.linalg.norm(embedding)
            student_id = "Unknown"
            name = "Unknown"
            course = ""
            subjects = ""
            role_label = "Unknown"
            max_score = 0

            # Compare with students
            for sid, info in face_db.items():
                score = cosine_similarity(embedding, info["embedding"])
                if score > max_score and score > 0.55:
                    max_score = score
                    student_id = info["student_id"]
                    name = info["name"]
                    course = info["course"]
                    subjects = info["subjects"]
                    role_label = "Student"

            # Compare visitors
            if student_id == "Unknown":
                for visitor in visitors_db:
                    score = cosine_similarity(embedding, visitor["embedding"])
                    if score > max_score and score > 0.55:
                        if datetime.now() <= datetime.strptime(visitor["expiration"], "%Y-%m-%d %H:%M:%S"):
                            max_score = score
                            student_id = visitor["visitor_id"]
                            name = visitor["name"]
                            course = f"Visitor ({visitor['type']})"
                            subjects = f"Visitor ({visitor['type']})"
                            role_label = f"Visitor ({visitor['type']})"

            # Compare teachers
            if student_id == "Unknown":
                for teacher in teachers_db:
                    score = cosine_similarity(embedding, teacher["embedding"])
                    if score > max_score and score > 0.55:
                        max_score = score
                        student_id = teacher["teacher_id"]
                        name = teacher["name"]
                        course = f"Teacher ({teacher['department']})"
                        subjects = teacher["subjects"]
                        role_label = "Teacher"
            
            # --------------------------
            # RECOGNITION STABILITY
            # --------------------------
            recognition_memory[(cam_id, track_id)].append(student_id)

            stable_id = Counter(recognition_memory[(cam_id, track_id)]).most_common(1)[0][0]

            if stable_id != "Unknown":
                student_id = stable_id

                # restore name info if needed
                if student_id in face_db:
                    name = face_db[student_id]["name"]
                    course = face_db[student_id]["course"]
                    subjects = face_db[student_id]["subjects"]
                    role_label = "Student"
            
            # --------------------------
            # COUNT UNKNOWN PERSON
            # --------------------------
            if student_id == "Unknown":
                stats["unknown_person"] += 1

            # Convert subjects list -> string before DB operations
            if isinstance(subjects, list):
                subjects_str = ",".join(subjects)
            else:
                subjects_str = subjects

            # Save snapshot
            face_crop = frame[y1:y2, x1:x2]
            image_name = None
            if face_crop.size != 0:
                ts = time.strftime("%Y%m%d_%H%M%S")
                image_name = f"{student_id if student_id != 'Unknown' else 'unknown'}_{ts}.jpg"
                cv2.imwrite(os.path.join(SNAPSHOT_DIR, image_name), face_crop)

            # Logging & attendance (fixed subjects)
            if should_log(student_id):
                c.execute("""
                    INSERT INTO logs
                    (student_id, name, course, subjects, time_in, score, image)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    student_id,
                    name,
                    course,
                    subjects_str,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    float(max_score),
                    image_name
                ))

            if student_id != "Unknown" and subjects_selected != "None":
                student_subject_list = [s.strip() for s in subjects_str.split(",")]

                if subjects_selected not in student_subject_list:
                    status_text = f"Detected ({subjects_str})"
                else:
                    if not already_marked_today(c, student_id, subjects_str):
                        status_text = "Present"
                        c.execute("""
                            INSERT INTO attendance
                            (student_id, name, course, subjects, time_in, score, status, image)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            student_id,
                            name,
                            course,
                            subjects_str,
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            float(max_score),
                            status_text,
                            image_name
                        ))
                    else:
                        status_text = "Present"
                        c.execute("""
                            UPDATE attendance
                            SET status = ?, score = ?, image = ?
                            WHERE student_id = ? AND DATE(time_in) = ?
                        """, (
                            status_text,
                            float(max_score),
                            image_name,
                            student_id,
                            datetime.now().strftime("%Y-%m-%d")
                        ))

            elif subjects_selected == "None":
                status_text = "No Subject Selected"

            else:
                status_text = "Detected Only"

            # --------------------------
            # YAWN DETECTION
            # --------------------------
            yawn_status = "Not Yawning"
            if face_crop.size != 0:
                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_rgb)
                
                # --- FIX: Ensure monotonically increasing timestamp ---
                ts_now = int(time.time() * 1000)
                timestamp = max(ts_now, last_timestamps[cam_id] + 1)
                last_timestamps[cam_id] = timestamp
                landmarkers[cam_id].detect_async(mp_image, timestamp)
                
                mar = 0
                result = latest_landmarks.get(cam_id)
                if result and result.face_landmarks:
                    landmarks = result.face_landmarks[0]
                    top = landmarks[13]
                    bottom = landmarks[14]
                    left = landmarks[78]
                    right = landmarks[308]
                    vertical = ((top.x - bottom.x)**2 + (top.y - bottom.y)**2)**0.5
                    horizontal = ((left.x - right.x)**2 + (left.y - right.y)**2)**0.5
                    mar = vertical / horizontal if horizontal != 0 else 0
                    if student_id not in yawning_buffer:
                        yawning_buffer[student_id] = deque(maxlen=MAR_FRAMES)
                        yawn_counts[student_id] = 0
                    yawning_buffer[student_id].append(mar)
                    if len(yawning_buffer[student_id]) == MAR_FRAMES:
                        avg_mar = sum(yawning_buffer[student_id]) / MAR_FRAMES
                        if avg_mar > MAR_THRESHOLD:
                            yawn_counts[student_id] += 1
                            yawning_buffer[student_id].clear()
                            yawn_status = "Yawning"
                            stats['yawning_count'] += 1
                        else:
                            yawn_status = "Not Yawning"

            # --------------------------
            # Draw face annotations (original)
            # --------------------------
            color = (0,255,0) if student_id != "Unknown" else (0,0,255)
            display_name = name if student_id != "Unknown" else "Unknown"

            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame, role_label, (x1,y1-30),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)
            cv2.putText(frame,f"{display_name} ({max_score:.2f})",
                        (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)
            cv2.putText(frame,
                        f"Yawning Status: {yawn_status}",
                        (x1,y2+40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0,255,255),
                        2)
            cv2.putText(frame,
                        status_text,
                        (x1,y2+20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0,255,255),
                        2)
            cv2.putText(
                frame,
                f"FPS: {fps}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        # --------------------------
        # DRAW SKELETONS & GESTURES ON WHOLE FRAME
        # --------------------------
        for result in pose_results:
            if result.keypoints is None:
                continue
            xy_pose = result.keypoints.xy.cpu().numpy()
            confs_pose = result.keypoints.conf.cpu().numpy()
            for person_idx in range(xy_pose.shape[0]):
                left_shoulder = xy_pose[person_idx,5]
                right_shoulder = xy_pose[person_idx,6]
                neck = ((left_shoulder[0]+right_shoulder[0])/2, (left_shoulder[1]+right_shoulder[1])/2)

                skeleton_color = (255,255,0)
                gesture_texts = []

                # Hands
                left_wrist = xy_pose[person_idx,9]
                right_wrist = xy_pose[person_idx,10]

                # --- Violating gestures detection ---
                # Sleeping (LESS STRICT)

                head_forward = xy_pose[person_idx,0,1] > (left_shoulder[1]+right_shoulder[1])/2 + 20
                hands_near_head = (
                    np.linalg.norm(left_wrist-xy_pose[person_idx,0]) < 100 or
                    np.linalg.norm(right_wrist-xy_pose[person_idx,0]) < 100
                )

                if head_forward and hands_near_head:
                    gesture_texts.append("Sleeping")
                    stats['violation_count'] += 1

                # Hand raised
                if left_wrist[1] < left_shoulder[1]-20:
                    gesture_texts.append("Left Hand Raised")
                if right_wrist[1] < right_shoulder[1]-20:
                    gesture_texts.append("Right Hand Raised")

                # Using phone
                for hand_idx, hx, hy in [(9,left_wrist[0],left_wrist[1]),(10,right_wrist[0],right_wrist[1])]:
                    for (px1,py1,px2,py2), _ in phone_boxes:
                        px, py = (px1+px2)/2,(py1+py2)/2
                        if np.sqrt((hx-px)**2 + (hy-py)**2) < HAND_PHONE_MAX_DIST:
                            gesture_texts.append("Using Phone")
                            stats['violation_count'] += 1
                            break

                # Color red if any violating gesture
                if len(gesture_texts)>0:
                    skeleton_color = (0,0,255)

                # Draw skeleton
                for kp1,kp2 in BONES:
                    if confs_pose[person_idx,kp1]>0.3 and confs_pose[person_idx,kp2]>0.3:
                        x1,y1 = xy_pose[person_idx,kp1]
                        x2,y2 = xy_pose[person_idx,kp2]
                        cv2.line(frame,(int(x1),int(y1)),(int(x2),int(y2)),skeleton_color,3)

                # Nose to neck
                if confs_pose[person_idx,0]>0.3:
                    cv2.line(frame,(int(xy_pose[person_idx,0,0]),int(xy_pose[person_idx,0,1])),
                             (int(neck[0]),int(neck[1])),skeleton_color,3)

                # Neck to shoulders
                if confs_pose[person_idx,5]>0.3:
                    cv2.line(frame,(int(neck[0]),int(neck[1])),
                             (int(left_shoulder[0]),int(left_shoulder[1])),skeleton_color,3)
                if confs_pose[person_idx,6]>0.3:
                    cv2.line(frame,(int(neck[0]),int(neck[1])),
                             (int(right_shoulder[0]),int(right_shoulder[1])),skeleton_color,3)

                # Draw joints
                for kp_idx in range(xy_pose.shape[1]):
                    if confs_pose[person_idx,kp_idx]>0.3:
                        x,y = xy_pose[person_idx,kp_idx]
                        cv2.circle(frame,(int(x),int(y)),5,(0,0,255),-1)
                cv2.circle(frame,(int(neck[0]),int(neck[1])),5,(0,255,255),-1)

                # Display gestures above neck
                y_offset = 0
                for text in gesture_texts:
                    cv2.putText(frame, text, (int(neck[0]), int(neck[1])-10-y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                    y_offset += 20

        conn.commit()
        conn.close()


        ret, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               frame_bytes +
               b'\r\n')

    cap.release()

current_student = {}
latest_face = None

# =========================
# LOG COOLDOWN (ANTI-SPAM)
# =========================
last_logged = {}
LOG_COOLDOWN = 10  # seconds

def should_log(student_id):
    now = time.time()
    if student_id not in last_logged:
        last_logged[student_id] = now
        return True
    if now - last_logged[student_id] >= LOG_COOLDOWN:
        last_logged[student_id] = now
        return True
    return False

def already_marked_today(c, student_id, subjects):

    today = datetime.now().strftime("%Y-%m-%d")

    c.execute("""
        SELECT 1 FROM attendance
        WHERE student_id=? AND subjects=? AND DATE(time_in)=?
    """, (student_id, subjects, today))

    return c.fetchone() is not None

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# =========================
# DASHBOARD
# =========================
@app.route("/")
def index():

    if "role" not in session:
        return redirect("/login")
    
    auto_delete_expired_visitors()  # 🔴 AUTO DELETE HERE
    conn = sqlite3.connect(LOG_DB)
    c = conn.cursor()

    # Total visitors
    c.execute("SELECT COUNT(*) FROM visitors")
    total_visitors = c.fetchone()[0]

    # Total teachers
    c.execute("SELECT COUNT(*) FROM teachers")
    total_teachers = c.fetchone()[0]

    # Total subjects 🔹 NEW
    c.execute("SELECT COUNT(*) FROM subjects")
    total_subjects = c.fetchone()[0]

    # Attendance per day
    c.execute("""
        SELECT DATE(time_in), COUNT(DISTINCT student_id)
        FROM attendance
        WHERE student_id != 'Unknown'
        GROUP BY DATE(time_in)
        ORDER BY DATE(time_in)
    """)
    rows = c.fetchall()
    conn.close()

    attendance_dates = [r[0] for r in rows]
    attendance_counts = [r[1] for r in rows]

    return render_template(
        "teacher_dashboard.html",
        role=session["role"],   # 🔴 important
        total_faces=len(face_db),          # Students
        total_visitors=total_visitors,     # Visitors
        total_teachers=total_teachers,     # Teachers
        total_subjects=total_subjects,     # 🔹 Subjects
        attendance_dates=json.dumps(attendance_dates),
        attendance_counts=json.dumps(attendance_counts)
    )

selected_face_id = None
latest_faces = []
latest_face = None

# store position of selected face to prevent switching
selected_face_bbox = None

@app.route("/select_face/<int:fid>")
def select_face(fid):
    global selected_face_id, latest_faces, selected_face_bbox

    selected_face_id = fid

    # save bbox of selected face
    if fid < len(latest_faces):
        selected_face_bbox = latest_faces[fid].bbox

    return "Face Selected"

# =========================
# Registration Stream (Camera Feed)
# =========================
def register_stream(cam_id=0):

    global latest_face, latest_faces, selected_face_id, selected_face_bbox

    frame_count = 0
    DETECT_EVERY_N = 2   # 🔥 skip frames (adjust: 2–4)
    SCALE = 0.5          # 🔥 resize for faster detection

    prev_time = 0

    while True:

        with frame_locks[cam_id]:
            if not frame_buffers[cam_id]:
                time.sleep(0.01)
                continue
            frame = frame_buffers[cam_id][-1].copy()

        # =====================
        # FPS COMPUTE (FIXED)
        # =====================
        curr_time = time.time()
        diff = curr_time - prev_time

        # جلوگیری division by zero
        if diff <= 0:
            diff = 1e-6  # very small number

        fps = 1 / diff
        prev_time = curr_time

        # =========================
        # Resize for speed
        # =========================
        small_frame = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)

        # =========================
        # Frame skipping
        # =========================
        if frame_count % DETECT_EVERY_N == 0:
            faces = face_app.get(small_frame)

            # Scale back bbox to original size
            for face in faces:
                face.bbox = [
                    int(coord / SCALE) for coord in face.bbox
                ]

            latest_faces = faces
        else:
            faces = latest_faces  # reuse previous result

        frame_count += 1
        latest_face = None

        # =========================
        # Stable face selection (UNCHANGED)
        # =========================
        if selected_face_bbox is not None and len(faces) > 0:

            prev_x1, prev_y1, prev_x2, prev_y2 = selected_face_bbox
            prev_cx = (prev_x1 + prev_x2) / 2
            prev_cy = (prev_y1 + prev_y2) / 2

            best_index = None
            best_distance = 999999

            for i, face in enumerate(faces):

                x1, y1, x2, y2 = face.bbox
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                dist = ((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) ** 0.5

                if dist < best_distance:
                    best_distance = dist
                    best_index = i

            if best_index is not None:
                selected_face_id = best_index
                selected_face_bbox = faces[best_index].bbox

        # If a face is selected manually
        if selected_face_id is not None and selected_face_id < len(faces):
            latest_face = faces[selected_face_id]

        # =========================
        # Draw bounding boxes
        # =========================
        for i, face in enumerate(faces):

            x1, y1, x2, y2 = map(int, face.bbox)

            color = (0, 0, 255)

            if selected_face_id == i:
                color = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(frame, f"ID:{i}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)
            
        # =====================
        # FPS DISPLAY
        # =====================
        cv2.putText(frame, f"FPS: {int(fps)}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

        # =========================
        # Encode frame (optimized)
        # =========================
        ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

# =========================
# REGISTER STUDENT
# =========================
@app.route("/register", methods=["GET", "POST"])
def register():
    global current_student

    conn = sqlite3.connect(LOG_DB)
    c = conn.cursor()

    c.execute("SELECT * FROM subjects ORDER BY subjects_name")
    subjects_list = c.fetchall()

    if request.method == "POST":

        student_id = request.form["student_id"].strip()
        name = request.form["name"].strip()
        course = request.form["course"].strip()

        # ✅ get multiple subjects
        subjects_ids = request.form.getlist("subjects[]")

        if not subjects_ids:
            conn.close()
            return render_template(
                "register.html",
                subjects_list=subjects_list,
                error="Please select at least one subject."
            )

        subjects_names = []

        # Get subjects names from IDs
        for subj_id in subjects_ids:
            c.execute("SELECT subjects_name FROM subjects WHERE id=?", (subj_id,))
            result = c.fetchone()

            if result is None:
                conn.close()
                return render_template(
                    "register.html",
                    subjects_list=subjects_list,
                    error="Invalid subject selected."
                )

            subjects_names.append(result[0])

        # Prevent duplicate student IDs
        if student_id in face_db:
            conn.close()
            return render_template(
                "register.html",
                subjects_list=subjects_list,
                error=f"Student ID {student_id} already exists!"
            )

        # ✅ store multiple subjects
        current_student = {
            "student_id": student_id,
            "name": name,
            "course": course,
            "subjects": subjects_names  # list of subjects
        }

        conn.close()
        return redirect(url_for("register_camera"))

    conn.close()
    return render_template("register.html", subjects_list=subjects_list)

@app.route("/register_camera")
def register_camera():
    return render_template("register_camera.html")

@app.route("/register_feed")
def register_feed():
    return Response(register_stream(cam_id=0),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# =========================
# TEMP STORAGE FOR MULTIPLE CAPTURES
# =========================
registration_embeddings = {"current": []}  # store embeddings before saving

# =========================
# Save Face Route
# =========================
@app.route("/save_face", methods=["POST"])
def save_face():
    global latest_face, current_student, registration_embeddings

    if latest_face is None or not current_student:
        return redirect(url_for("register_camera"))

    student_id = current_student["student_id"]

    # Initialize buffer if not exists
    if "current" not in registration_embeddings:
        registration_embeddings["current"] = []

    # Append current embedding
    registration_embeddings["current"].append(latest_face.embedding)
    print(f"[INFO] Captured {len(registration_embeddings['current'])}/5 frames for {student_id}")

    # Only save after 10 captures
    if len(registration_embeddings["current"]) >= 10:
        avg_embedding = np.mean(registration_embeddings["current"], axis=0)

        # Save in face_db
        face_db[student_id] = {
            "student_id": student_id,
            "name": current_student["name"],
            "course": current_student["course"],
            "subjects": current_student["subjects"],
            "embedding": avg_embedding
        }

        # Save face snapshot
        x1, y1, x2, y2 = map(int, latest_face.bbox)
        face_crop = latest_face.normed_face if hasattr(latest_face, "normed_face") else None
        if face_crop is not None and face_crop.size != 0:
            ts = time.strftime("%Y%m%d_%H%M%S")
            image_name = f"{student_id}_{ts}.jpg"
            cv2.imwrite(os.path.join(SNAPSHOT_DIR, image_name), face_crop)

            # Log registration
            conn = sqlite3.connect(LOG_DB)
            c = conn.cursor()
            c.execute(
                "INSERT INTO logs (student_id, name, course, subjects, time_in, score, image) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    student_id,
                    current_student["name"],
                    current_student["course"],
                    current_student["subjects"],
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    1.0,
                    image_name
                )
            )
            # Attendance
            c.execute(
                "INSERT INTO attendance (student_id, name, course, subjects, time_in, score, status, image) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    student_id,
                    current_student["name"],
                    current_student["course"],
                    current_student["subjects"],
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    1.0,
                    "Present",
                    image_name
                )
            )
            conn.commit()
            conn.close()

        # Save DB
        with open(DB_PATH, "wb") as f:
            pickle.dump(face_db, f)

        # Clear embeddings buffer
        registration_embeddings["current"].clear()
        print(f"[SUCCESS] Saved stable embedding for {student_id}")

    return redirect(url_for("register_camera"))


@app.route("/quit")
def quit_camera():
    return redirect(url_for("index"))

# =========================
# FACE RECOGNITION
# =========================
def recognize_stream():
    auto_delete_expired_visitors()
    while True:
        with frame_locks[cam_id]:
            if not frame_buffers[cam_id]:
                time.sleep(0.05)
                continue
            frame = frame_buffers[cam_id][-1].copy()

        faces = face_app.get(frame)
        for face in faces:
            embedding = face.embedding
            student_id = "Unknown"
            name = "Unknown"
            course = ""
            subjects = ""
            role_label = "Unknown"
            max_score = 0

            # Compare with known faces
            for sid, info in face_db.items():
                score = cosine_similarity(embedding, info["embedding"])
                if score > max_score and score > 0.55:
                    max_score = score
                    student_id = info["student_id"]
                    name = info["name"]
                    course = info["course"]
                    subjects = info["subjects"]
                    role_label = "Student"

            # Check visitors if not found
            if student_id == "Unknown":
                stats["unknown_person"] = stats.get("unknown_person", 0) + 1
                conn = sqlite3.connect(LOG_DB)
                c = conn.cursor()
                c.execute("SELECT visitor_id, name, visitor_type, expiration_date, embedding FROM visitors")
                for row in c.fetchall():
                    v_id, v_name, v_type, exp, v_embedding = row
                    v_embedding = pickle.loads(v_embedding)
                    score = cosine_similarity(embedding, v_embedding)
                    if score > max_score and score > 0.55:
                        # Check expiration
                        if datetime.now() <= datetime.strptime(exp, "%Y-%m-%d %H:%M:%S"):
                            max_score = score
                            student_id = v_id
                            name = v_name
                            course = f"Visitor ({v_type})"
                            subjects = f"Visitor ({v_type})"
                            role_label = f"Visitor ({v_type})"
                conn.close()

            # Save snapshot for both known and unknown faces
            x1, y1, x2, y2 = map(int, face.bbox)
            face_crop = frame[y1:y2, x1:x2]
            image_name = None
            if face_crop.size != 0:
                ts = time.strftime("%Y%m%d_%H%M%S")
                if student_id != "Unknown":
                    image_name = f"{student_id}_{ts}.jpg"
                else:
                    image_name = f"unknown_{ts}.jpg"
                cv2.imwrite(os.path.join(SNAPSHOT_DIR, image_name), face_crop)

            # Single log entry
            if should_log(student_id):
                conn = sqlite3.connect(LOG_DB)
                c = conn.cursor()
                c.execute(
                    "INSERT INTO logs (student_id, name, course, subjects, time_in, score, image) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        student_id,
                        name,
                        course,
                        subjects,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        float(max_score),
                        image_name
                    )
                )
                # Also insert into attendance
                if student_id != "Unknown":
                    c.execute(
                        "INSERT INTO attendance (student_id, name, course, subjects, time_in, score, status, image) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            student_id,
                            name,
                            course,
                            subjects,
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            float(max_score),
                            "Present",
                            image_name
                        )
                    )
                conn.commit()
                conn.close()

            color = (0, 255, 0) if student_id != "Unknown" else (0, 0, 255)
            display_name = name if student_id != "Unknown" else "Unknown"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # Role label (TOP)
            cv2.putText(
                frame,
                role_label,
                (x1, y1 - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
            # 🔹 Draw role label (Student or Visitor)
            role_label = "Student" if "Visitor" not in course else course
            role_color = (255, 255, 0)  # yellow
            cv2.putText(
                frame,
                role_label,
                (x1, y1 - 30),          # above the name
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                role_color,
                2
            )            
            cv2.putText(
                frame,
                f"{display_name} ({max_score:.2f})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

@app.route("/recognize")
def recognize():
    return render_template("recognize.html")

@app.route('/recognize_feed/<int:cam_id>')
def recognize_feed(cam_id):
    return Response(
        generate_frames(cam_id=cam_id),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# =========================
# LOGS & ATTENDANCE
# =========================
@app.route("/logs")
def logs():

    conn = sqlite3.connect(LOG_DB)
    conn.row_factory = sqlite3.Row   # ✅ allows log["column_name"]
    c = conn.cursor()

    c.execute("""
        SELECT 
            id,
            student_id,
            name,
            course,
            subjects,
            time_in,
            score,
            image
        FROM logs
        ORDER BY id DESC
    """)

    logs_data = c.fetchall()

    conn.close()

    return render_template("logs.html", logs=logs_data)

@app.route("/attendance")
def attendance():
    conn = sqlite3.connect(LOG_DB)
    c = conn.cursor()

    # 1️⃣ Get all subjects for the filter dropdown
    c.execute("SELECT subjects_name FROM subjects ORDER BY subjects_name")
    subjects = [row[0] for row in c.fetchall()]

    # 2️⃣ Get all registered students (from face_db)
    students = []
    for sid, info in face_db.items():
        students.append({
            "student_id": sid,
            "name": info["name"],
            "subjects": info.get("subjects", []),  # assumes each student has a list of subjects
            "attendance": {}  # will fill later
        })

    # 3️⃣ Get selected month/year/subject (default = current month/year, all subjects)
    month = int(request.args.get("month", datetime.now().month))
    year = int(request.args.get("year", datetime.now().year))
    selected_subject = request.args.get("subject", "")

    first_day = datetime(year, month, 1)
    last_day = datetime(year, month, calendar.monthrange(year, month)[1])

    # 4️⃣ Fetch attendance logs
    c.execute("""
        SELECT
            student_id,
            SUBSTR(time_in, 1, 10) as day,
            status
        FROM attendance
        WHERE SUBSTR(time_in, 1, 10) BETWEEN ? AND ?
    """, (
        first_day.strftime("%Y-%m-%d"),
        last_day.strftime("%Y-%m-%d")
    ))

    logs = c.fetchall()
    conn.close()

    # 5️⃣ Build lookup dict
    log_dict = defaultdict(lambda: defaultdict(lambda: "Absent"))
    for student_id, day, status in logs:
        log_dict[student_id][day] = status

    # 6️⃣ Generate month dates (weekdays only)
    month_dates = []
    current = first_day
    while current <= last_day:
        if current.weekday() < 5:  # Mon-Fri
            month_dates.append(current)
        current += timedelta(days=1)

    # 7️⃣ Fill attendance per student
    for student in students:
        for date in month_dates:
            day_str = date.strftime("%Y-%m-%d")
            student["attendance"][day_str] = log_dict.get(student["student_id"], {}).get(day_str, "Absent")

    # 8️⃣ Filter students by selected subject if any
    if selected_subject:
        students = [s for s in students if selected_subject in s.get("subjects", [])]

    return render_template(
        "attendance.html",
        attendance=students,
        month_dates=month_dates,
        now=first_day,
        selected_month=month,
        selected_year=year,
        subjects=subjects,            # ✅ pass subject list
        selected_subject=selected_subject,  # ✅ pass currently selected subject
        datetime=datetime
    )

@app.route("/teacher_attendance")
def teacher_attendance():
    conn = sqlite3.connect(LOG_DB)
    c = conn.cursor()

    # 1️⃣ Get all subjects for filter dropdown
    c.execute("SELECT subjects_name FROM subjects ORDER BY subjects_name")
    subjects = [row[0] for row in c.fetchall()]

    # 2️⃣ Get all registered teachers
    c.execute("""
        SELECT teacher_id, name, subjects
        FROM teachers
    """)
    teacher_rows = c.fetchall()

    teachers = []
    for t_id, name, t_subjects in teacher_rows:
        teachers.append({
            "teacher_id": t_id,
            "name": name,
            "subjects": t_subjects.split(",") if t_subjects else [],
            "attendance": {}
        })

    # 3️⃣ Get selected month/year/subject
    month = int(request.args.get("month", datetime.now().month))
    year = int(request.args.get("year", datetime.now().year))
    selected_subject = request.args.get("subject", "")

    first_day = datetime(year, month, 1)
    last_day = datetime(year, month, calendar.monthrange(year, month)[1])

    # 4️⃣ Fetch teacher attendance logs
    c.execute("""
        SELECT
            student_id,
            SUBSTR(time_in, 1, 10) as day,
            status,
            subjects
        FROM attendance
        WHERE SUBSTR(time_in, 1, 10) BETWEEN ? AND ?
    """, (
        first_day.strftime("%Y-%m-%d"),
        last_day.strftime("%Y-%m-%d")
    ))

    logs = c.fetchall()
    conn.close()

    # 5️⃣ Build lookup dict (teacher_id used as student_id in attendance table)
    log_dict = defaultdict(dict)

    for person_id, day, status, subject in logs:
        for teacher in teachers:
            if teacher["teacher_id"] == person_id:
                # If subject filter selected, only include matching subject
                if selected_subject:
                    if selected_subject in (subject.split(",") if subject else []):
                        log_dict[person_id][day] = status
                else:
                    log_dict[person_id][day] = status

    # 6️⃣ Generate month weekdays (Mon-Fri only)
    month_dates = []
    current = first_day
    while current <= last_day:
        if current.weekday() < 5:
            month_dates.append(current)
        current += timedelta(days=1)

    # 7️⃣ Fill attendance per teacher
    today = datetime.now().date()

    for teacher in teachers:
        for date in month_dates:
            day_str = date.strftime("%Y-%m-%d")

            if date.date() > today:
                status = ""   # future → blank

            elif day_str in log_dict.get(teacher["teacher_id"], {}):
                status = log_dict[teacher["teacher_id"]][day_str]  # actual record

            else:
                status = "Absent"   # past but no record → Absent

            teacher["attendance"][day_str] = status

    # 8️⃣ Filter teachers by selected subject if any
    if selected_subject:
        teachers = [
            t for t in teachers
            if selected_subject in t.get("subjects", [])
        ]

    # 9️⃣ Pass user role to template
    user_role = session.get("role", "teacher")  # default to Teacher if not logged in
    print("[DEBUG] Current Role:", user_role)

    return render_template(
        "teacher_attendance.html",
        attendance=teachers,
        month_dates=month_dates,
        now=first_day,
        selected_month=month,
        selected_year=year,
        subjects=subjects,
        selected_subject=selected_subject,
        datetime=datetime,
        user_role=user_role  # ✅ Added this line
    )

# =========================
# SERVE SNAPSHOTS
# =========================
@app.route("/face_snapshots/<filename>")
def face_snapshot(filename):
    return send_from_directory(SNAPSHOT_DIR, filename)

# =========================
# MANAGE / EDIT / DELETE REGISTERED FACES
# =========================
@app.route("/manage_registered_students", methods=["GET", "POST"])
def manage_registered_students():
    global face_db

    if request.method == "POST":
        # -----------------------
        # 1️⃣ UPDATE / SAVE STUDENT
        # -----------------------
        edit_id = request.form.get("edit_id")
        new_student_id = request.form.get("new_student_id")
        new_name = request.form.get("new_name")
        new_course = request.form.get("new_course")
        # FIX HERE
        new_subjects_list = request.form.getlist("new_subjects[]")
        new_subjects = ", ".join(new_subjects_list)

        if edit_id and edit_id in face_db:
            face_db[new_student_id.strip()] = {
                "student_id": new_student_id.strip(),
                "name": new_name.strip(),
                "course": new_course.strip(),
                "subjects": new_subjects.strip(),
                "embedding": face_db.pop(edit_id)["embedding"]
            }

            conn = sqlite3.connect(LOG_DB)
            c = conn.cursor()
            c.execute("UPDATE attendance SET student_id=?, name=?, course=?, subjects=? WHERE student_id=?",
                      (new_student_id, new_name, new_course, new_subjects, edit_id))
            c.execute("UPDATE logs SET student_id=?, name=?, course=?, subjects=? WHERE student_id=?",
                      (new_student_id, new_name, new_course, new_subjects, edit_id))
            conn.commit()
            conn.close()

        # -----------------------
        # 2️⃣ DELETE STUDENT
        # -----------------------
        delete_id = request.form.get("delete_id")
        if delete_id and delete_id in face_db:
            del face_db[delete_id]

            conn = sqlite3.connect(LOG_DB)
            c = conn.cursor()
            c.execute("DELETE FROM attendance WHERE student_id=?", (delete_id,))
            conn.commit()
            conn.close()

        with open(DB_PATH, "wb") as f:
            pickle.dump(face_db, f)

        return redirect(url_for("manage_registered_students"))

    # ✅ LOAD SUBJECTS FROM DATABASE
    conn = sqlite3.connect(LOG_DB)
    c = conn.cursor()
    c.execute("SELECT * FROM subjects ORDER BY subjects_name")
    subjects_list = c.fetchall()
    conn.close()

    # ✅ PASS subjects_list TO TEMPLATE
    return render_template(
        "manage_registered_students.html",
        faces=list(face_db.values()),
        subjects_list=subjects_list
    )

current_visitor = {}

@app.route("/register_visitor", methods=["GET", "POST"])
def register_visitor():
    global current_visitor
    if request.method == "POST":
        visitor_id = request.form["visitor_id"].strip()
        name = request.form["name"].strip()
        visitor_type = request.form["visitor_type"]
        purpose = request.form["purpose"].strip()

        conn = sqlite3.connect(LOG_DB)
        c = conn.cursor()
        c.execute("SELECT * FROM visitors WHERE visitor_id=?", (visitor_id,))
        if c.fetchone():
            conn.close()
            return render_template("register_visitor.html", error="Visitor ID already exists!")
        conn.close()

        current_visitor = {
            "visitor_id": visitor_id,
            "name": name,
            "visitor_type": visitor_type,
            "purpose": purpose
        }
        return redirect(url_for("register_visitor_camera"))
    return render_template("register_visitor.html")

@app.route("/register_visitor_camera")
def register_visitor_camera():
    return render_template("register_visitor_camera.html")

@app.route("/register_visitor_feed")
def register_visitor_feed():
    return Response(register_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

# =========================
# Visitor multi-frame embedding
# =========================
visitor_embeddings = {"current": []}  # temporary buffer for current visitor

@app.route("/save_visitor_face", methods=["POST"])
def save_visitor_face():
    global latest_face, current_visitor

    if latest_face is not None and current_visitor:
        # Add latest embedding to buffer
        visitor_embeddings["current"].append(latest_face.embedding)

        # Wait until we have 10 frames
        if len(visitor_embeddings["current"]) < 10:
            return "Frame captured, not yet enough frames.", 200

        # Average embeddings for stability
        avg_embedding = np.mean(visitor_embeddings["current"], axis=0)

        visitor_id = current_visitor["visitor_id"]
        expiration = calculate_expiration(current_visitor["visitor_type"])

        # Save visitor in DB
        conn = sqlite3.connect(LOG_DB)
        c = conn.cursor()
        c.execute("""
            INSERT INTO visitors (visitor_id, name, visitor_type, purpose, expiration_date, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            visitor_id,
            current_visitor["name"],
            current_visitor["visitor_type"],
            current_visitor["purpose"],
            expiration.strftime("%Y-%m-%d %H:%M:%S"),
            pickle.dumps(avg_embedding)
        ))
        conn.commit()
        conn.close()

        # Save snapshot of latest face
        x1, y1, x2, y2 = map(int, latest_face.bbox)
        face_crop = latest_face.normed_face if hasattr(latest_face, "normed_face") else None
        if face_crop is not None and face_crop.size != 0:
            ts = time.strftime("%Y%m%d_%H%M%S")
            image_name = f"visitor_{visitor_id}_{ts}.jpg"
            cv2.imwrite(os.path.join(SNAPSHOT_DIR, image_name), face_crop)

        # Clear buffer for next visitor
        visitor_embeddings["current"].clear()

        return "Visitor stable embedding saved ✅", 200

    return "No face detected", 400

@app.route("/manage_visitors", methods=["GET", "POST"])
def manage_visitors():
    conn = sqlite3.connect(LOG_DB)
    c = conn.cursor()

    # Edit visitor
    if request.method == "POST":
        edit_id = request.form.get("edit_id")
        new_name = request.form.get("new_name")
        new_type = request.form.get("new_type")
        new_purpose = request.form.get("new_purpose")

        if edit_id:
            c.execute("""
                UPDATE visitors SET name=?, visitor_type=?, purpose=? WHERE visitor_id=?
            """, (new_name, new_type, new_purpose, edit_id))

        delete_id = request.form.get("delete_id")
        if delete_id:
            c.execute("DELETE FROM visitors WHERE visitor_id=?", (delete_id,))
        conn.commit()

    c.execute("SELECT * FROM visitors")
    visitors = c.fetchall()
    conn.close()
    return render_template("manage_visitors.html", visitors=visitors)

@app.route("/download/attendance_csv")
def download_attendance_csv():
    conn = sqlite3.connect(LOG_DB)

    query = """
        SELECT
            student_id,
            name,
            course,
            subjects,
            time_in,
            status,
            score
        FROM attendance
        ORDER BY time_in DESC
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        return "No attendance data available", 204

    reports_dir = os.path.join(BASE_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    filename = f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    filepath = os.path.join(reports_dir, filename)

    df.to_csv(filepath, index=False)

    return send_file(filepath, as_attachment=True)

@app.route("/download/visitor_logs_csv")
def download_visitor_logs_csv():
    conn = sqlite3.connect(LOG_DB)

    query = """
        SELECT
            visitor_id,
            name,
            visitor_type,
            purpose,
            expiration_date
        FROM visitors
        ORDER BY expiration_date DESC
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        return "No visitor data available", 204

    reports_dir = os.path.join(BASE_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    filename = f"visitors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    filepath = os.path.join(reports_dir, filename)

    df.to_csv(filepath, index=False)

    return send_file(filepath, as_attachment=True)

@app.route("/update_attendance", methods=["POST"])
def update_attendance():
    data = request.get_json()
    student_id = data.get("student_id")
    date_str = data.get("date")  # format: "YYYY-MM-DD"
    status = data.get("status", "Absent")  # default to "Absent" if not provided

    try:
        conn = sqlite3.connect(LOG_DB)
        c = conn.cursor()

        # Check if record exists
        c.execute("""
            SELECT id FROM attendance
            WHERE student_id = ? AND DATE(time_in) = ?
        """, (student_id, date_str))
        row = c.fetchone()

        if row:
            # Update existing record
            c.execute("""
                UPDATE attendance
                SET status = ?
                WHERE id = ?
            """, (status, row[0]))
        else:
            # Insert new record with default values for unknown student
            c.execute("""
                INSERT INTO attendance
                (student_id, name, course, subjects, time_in, score, status, image)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                student_id,
                "Unknown",               # name
                "",                      # course
                "",                      # subjects
                f"{date_str} 08:00:00",  # time_in
                0,                       # score
                status,                  # status ("Absent")
                None                     # image
            ))

        conn.commit()
        conn.close()
        return {"success": True}

    except Exception as e:
        print("[ERROR] Update attendance:", e)
        return {"success": False, "message": str(e)}

@app.route("/subjects", methods=["GET", "POST"])
def subjects():
    conn = sqlite3.connect(LOG_DB)
    c = conn.cursor()

    if request.method == "POST":
        subjects_name = request.form.get("subjects_name")
        schedule = request.form.get("schedule")
        teacher_name = request.form.get("teacher_name")
        room_number = request.form.get("room_number")
        rows = request.form.get("rows")
        columns = request.form.get("columns")

        try:
            c.execute("""
                INSERT INTO subjects
                (subjects_name, schedule, teacher_name, room_number, rows, columns)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (subjects_name, schedule, teacher_name, room_number, rows, columns))

            conn.commit()

        except sqlite3.IntegrityError:
            pass

    c.execute("SELECT * FROM subjects ORDER BY subjects_name")
    subjects_list = c.fetchall()

    conn.close()

    return render_template("subjects.html", subjects_list=subjects_list)

@app.route("/delete_subjects/<int:id>")
def delete_subjects(id):

    global face_db

    conn = sqlite3.connect(LOG_DB)
    c = conn.cursor()

    c.execute("SELECT subjects_name FROM subjects WHERE id=?", (id,))
    row = c.fetchone()

    if row:
        subject_name = row[0]

        # =========================
        # 🔴 CHECK 1: Used by students
        # =========================
        for student_id, info in face_db.items():
            subjects_data = info.get("subjects", [])

            if isinstance(subjects_data, list):
                subjects_list = subjects_data
            else:
                subjects_list = [s.strip() for s in subjects_data.split(",") if s.strip()]

            if subject_name in subjects_list:
                conn.close()
                return f"❌ Cannot delete '{subject_name}' — still assigned to students."

        # =========================
        # 🔴 CHECK 2: Used in attendance
        # =========================
        c.execute("SELECT COUNT(*) FROM attendance WHERE subjects LIKE ?", (f"%{subject_name}%",))
        count = c.fetchone()[0]

        if count > 0:
            conn.close()
            return f"❌ Cannot delete '{subject_name}' — used in attendance records."

        # =========================
        # ✅ SAFE TO DELETE (your original code)
        # =========================

        for student_id, info in face_db.items():

            subjects_data = info.get("subjects", [])

            if isinstance(subjects_data, list):
                subjects_list = subjects_data
            else:
                subjects_list = [s.strip() for s in subjects_data.split(",") if s.strip()]

            if subject_name in subjects_list:
                subjects_list.remove(subject_name)

            face_db[student_id]["subjects"] = ", ".join(subjects_list)

        with open(DB_PATH, "wb") as f:
            pickle.dump(face_db, f)

        c.execute("DELETE FROM attendance WHERE subjects LIKE ?", (f"%{subject_name}%",))
        c.execute("DELETE FROM subjects WHERE id=?", (id,))
        conn.commit()

    conn.close()
    return redirect(url_for("subjects"))

@app.route("/edit_subjects/<int:id>", methods=["GET", "POST"])
def edit_subjects(id):

    conn = sqlite3.connect(LOG_DB)
    c = conn.cursor()

    if request.method == "POST":

        subjects_name = request.form.get("subjects_name")
        schedule = request.form.get("schedule")
        teacher_name = request.form.get("teacher_name")
        room_number = request.form.get("room_number")
        rows = request.form.get("rows")
        columns = request.form.get("columns")

        c.execute("""
            UPDATE subjects
            SET subjects_name=?, schedule=?, teacher_name=?, room_number=?, rows=?, columns=?
            WHERE id=?
        """, (subjects_name, schedule, teacher_name, room_number, rows, columns, id))

        conn.commit()

        return redirect(url_for("subjects"))

    c.execute("SELECT * FROM subjects WHERE id=?", (id,))
    subjects_data = c.fetchone()

    conn.close()

    return render_template("edit_subjects.html", subjects=subjects_data)

@app.route("/get_subjects")
def get_subjects():
    conn = sqlite3.connect(LOG_DB)
    c = conn.cursor()

    c.execute("SELECT * FROM subjects ORDER BY subjects_name")
    rows = c.fetchall()

    conn.close()

    return jsonify([
        {
            "id": row[0],
            "name": row[1]
        }
        for row in rows
    ])

@app.route("/get_attendance_by_subjects/<subjects>")
def get_attendance_by_subjects_route(subjects):
    attendance = get_attendance_by_subjects(subjects)
    return jsonify(attendance)

def get_attendance_by_subjects(subjects):
    conn = sqlite3.connect(LOG_DB)
    c = conn.cursor()
    c.execute("""
        SELECT student_id, name, course, subjects, time_in, status, image
        FROM attendance
        WHERE LOWER(subjects) = LOWER(?)
        ORDER BY time_in DESC
    """, (subjects,))
    rows = c.fetchall()
    conn.close()
    return [
        {
            "student_id": r[0],
            "name": r[1],
            "course": r[2],
            "subjects": r[3],
            "time_in": r[4],
            "status": r[5],
            "image": r[6]
        }
        for r in rows
    ]

@app.route("/api/noise_level", methods=["POST"])
def api_noise_level():
    data = request.json or {}
    noise_level = data.get('noise_level')
    if noise_level is None:
        return jsonify({'status': 'error', 'error': 'No noise level provided'}), 400

    # Update global stats
    stats['noise_level'] = noise_level

    return jsonify({'status': 'success', 'noise_level': stats['noise_level']})

@app.route("/api/behavior_stats", methods=["GET"])
def api_behavior_stats():
    """
    Returns the current behavior stats:
    - number of yawning people
    - number of people using phone
    - number of people with violation gestures
    - number of unknown person
    - noise level
    Also logs stats to the database.
    """
    global stats
    yawning = stats.get("yawning_count", 0)
    using_phone = stats.get("phone_count", 0)
    violation_gesture = stats.get("violation_count", 0)
    unknown_person = stats.get("unknown_person", 0)
    noise_level = stats.get("noise_level", 0)

    # 1️⃣ Save to DB
    try:
        conn = sqlite3.connect(LOG_DB)
        c = conn.cursor()
        c.execute("""
            INSERT INTO behavior_logs
            (timestamp, yawning_count, phone_count, violation_count, unknown_person_count, noise_level)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            yawning,
            using_phone,
            violation_gesture,
            unknown_person,
            noise_level
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print("[ERROR] Saving behavior stats:", e)

    # 2️⃣ Return JSON response
    response = {
        "yawning": yawning,
        "using_phone": using_phone,
        "violation_gesture": violation_gesture,
        "noise_level": noise_level,
        "unknown_person": unknown_person
    }
    return jsonify(response)

current_teacher = {}
teacher_embeddings = {"current": []}

@app.route("/register_teacher", methods=["GET", "POST"])
def register_teacher():
    global current_teacher

    conn = sqlite3.connect(LOG_DB)
    c = conn.cursor()

    # Fetch subjects list for form
    c.execute("SELECT * FROM subjects ORDER BY subjects_name")
    subjects_list = c.fetchall()

    if request.method == "POST":
        teacher_id = request.form["teacher_id"].strip()
        name = request.form["name"].strip()
        department = request.form["department"].strip()
        time_in = request.form["time_in"].strip()
        time_out = request.form["time_out"].strip()

        # ✅ get multiple subjects
        subjects_ids = request.form.getlist("subjects[]")
        if not subjects_ids:
            conn.close()
            return render_template(
                "register_teacher.html",
                subjects_list=subjects_list,
                error="Please select at least one subject."
            )

        subjects_names = []
        for subj_id in subjects_ids:
            c.execute("SELECT subjects_name FROM subjects WHERE id=?", (subj_id,))
            result = c.fetchone()
            if result is None:
                conn.close()
                return render_template(
                    "register_teacher.html",
                    subjects_list=subjects_list,
                    error="Invalid subject selected."
                )
            subjects_names.append(result[0])

        # Check for duplicate teacher ID
        c.execute("SELECT * FROM teachers WHERE teacher_id=?", (teacher_id,))
        if c.fetchone():
            conn.close()
            return render_template(
                "register_teacher.html",
                subjects_list=subjects_list,
                error=f"Teacher ID {teacher_id} already exists!"
            )

        # Store current teacher info
        current_teacher = {
            "teacher_id": teacher_id,
            "name": name,
            "department": department,
            "subjects": subjects_names,  # list of subjects
            "time_in": time_in,
            "time_out": time_out
        }

        conn.close()
        return redirect(url_for("register_teacher_camera"))

    conn.close()
    return render_template("register_teacher.html", subjects_list=subjects_list)

@app.route("/register_teacher_camera")
def register_teacher_camera():
    return render_template("register_teacher_camera.html")

@app.route("/register_teacher_feed")
def register_teacher_feed():
    return Response(register_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/save_teacher_face", methods=["POST"])
def save_teacher_face():
    global latest_face, current_teacher, teacher_embeddings

    if latest_face is not None and current_teacher:
        teacher_embeddings["current"].append(latest_face.embedding)

        if len(teacher_embeddings["current"]) < 10:
            return "Frame captured, not enough frames yet.", 200

        avg_embedding = np.mean(teacher_embeddings["current"], axis=0)

        teacher_id = current_teacher["teacher_id"]

        # ✅ Convert subjects list to comma-separated string
        subjects_str = ",".join(current_teacher["subjects"])

        # Save teacher in DB
        conn = sqlite3.connect(LOG_DB)
        c = conn.cursor()
        c.execute("""
            INSERT INTO teachers (teacher_id, name, department, subjects, time_in, time_out, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            teacher_id,
            current_teacher["name"],
            current_teacher["department"],
            subjects_str,   # ✅ subjects list -> string
            current_teacher["time_in"],
            current_teacher["time_out"],
            pickle.dumps(avg_embedding)
        ))
        conn.commit()
        conn.close()

        # Save snapshot
        x1, y1, x2, y2 = map(int, latest_face.bbox)
        face_crop = latest_face.normed_face if hasattr(latest_face, "normed_face") else None
        if face_crop is not None and face_crop.size != 0:
            ts = time.strftime("%Y%m%d_%H%M%S")
            image_name = f"teacher_{teacher_id}_{ts}.jpg"
            cv2.imwrite(os.path.join(SNAPSHOT_DIR, image_name), face_crop)

        teacher_embeddings["current"].clear()
        return "Teacher stable embedding saved ✅", 200

    return "No face detected", 400

@app.route("/manage_teachers", methods=["GET", "POST"])
def manage_teachers():
    conn = sqlite3.connect(LOG_DB)
    c = conn.cursor()

    if request.method == "POST":
        edit_id = request.form.get("edit_id")
        new_name = request.form.get("new_name")
        new_department = request.form.get("new_department")

        # ✅ FIX: get multiple checkbox values correctly
        new_subjects = request.form.getlist("new_subjects[]")

        time_in = request.form.get("time_in")
        time_out = request.form.get("time_out")

        # convert list → string for DB storage
        subjects_str = ", ".join(new_subjects)

        if edit_id:
            c.execute("""
                UPDATE teachers
                SET name=?, department=?, subjects=?, time_in=?, time_out=?
                WHERE teacher_id=?
            """, (new_name, new_department, subjects_str, time_in, time_out, edit_id))

        delete_id = request.form.get("delete_id")
        if delete_id:
            c.execute("DELETE FROM teachers WHERE teacher_id=?", (delete_id,))

        conn.commit()

    c.execute("SELECT * FROM teachers ORDER BY name")
    teachers = c.fetchall()
    conn.close()

    # load subjects
    conn = sqlite3.connect(LOG_DB)
    c = conn.cursor()
    c.execute("SELECT * FROM subjects ORDER BY subjects_name")
    subjects_list = c.fetchall()
    conn.close()

    return render_template(
        "manage_teachers.html",
        teachers=teachers,
        subjects_list=subjects_list
    )

@app.route("/behavior_logs")
def behavior_logs():
    conn = sqlite3.connect(LOG_DB)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM behavior_logs ORDER BY id DESC")
    logs = c.fetchall()
    conn.close()
    return render_template("behavior_logs.html", logs=logs)

@app.route("/download_behavior_csv")
def download_behavior_csv():

    conn = sqlite3.connect(LOG_DB)
    c = conn.cursor()
    c.execute("SELECT * FROM behavior_logs ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow([
        "ID",
        "Timestamp",
        "Yawning Count",
        "Phone Usage Count",
        "Violation Count",
        "Unknown Person Count",
        "Noise Level"
    ])

    # Data rows
    writer.writerows(rows)

    output.seek(0)

    return Response(
        output,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=behavior_logs.csv"}
    )

@app.route("/login", methods=["GET","POST"])
def login():

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]
        role = request.form["role"]
        session["system_id"] = SYSTEM_ID

        conn = sqlite3.connect(DB_LOGIN)
        c = conn.cursor()

        c.execute(
            "SELECT role FROM users WHERE username=? AND password=? AND role=?",
            (username, password, role)
        )

        user = c.fetchone()
        conn.close()

        if user:
            session["username"] = username
            session["role"] = role

            if role == "admin":
                return redirect("/admin_dashboard")

            if role == "teacher":
                return redirect("/teacher_dashboard")

        return render_template("login.html", error="Invalid credentials. Please try again.")

    return render_template("login.html")

@app.route("/admin_dashboard")
def admin_dashboard():
    if session.get("role") != "admin":
        return "Unauthorized"
    
    auto_delete_expired_visitors()  # 🔴 AUTO DELETE HERE
    conn = sqlite3.connect(LOG_DB)
    c = conn.cursor()

    # Total visitors
    c.execute("SELECT COUNT(*) FROM visitors")
    total_visitors = c.fetchone()[0]

    # Total teachers
    c.execute("SELECT COUNT(*) FROM teachers")
    total_teachers = c.fetchone()[0]

    # Total subjects 🔹 NEW
    c.execute("SELECT COUNT(*) FROM subjects")
    total_subjects = c.fetchone()[0]

    # Attendance per day
    c.execute("""
        SELECT DATE(time_in), COUNT(DISTINCT student_id)
        FROM attendance
        WHERE student_id != 'Unknown'
        GROUP BY DATE(time_in)
        ORDER BY DATE(time_in)
    """)
    rows = c.fetchall()
    conn.close()

    attendance_dates = [r[0] for r in rows]
    attendance_counts = [r[1] for r in rows]

    return render_template(
        "admin_dashboard.html",
        role=session["role"],   # 🔴 important
        total_faces=len(face_db),          # Students
        total_visitors=total_visitors,     # Visitors
        total_teachers=total_teachers,     # Teachers
        total_subjects=total_subjects,     # 🔹 Subjects
        attendance_dates=json.dumps(attendance_dates),
        attendance_counts=json.dumps(attendance_counts))

@app.route("/logout")
def logout():
    session.clear()  # Clear all session data
    return redirect("/login")

@app.route("/teacher_dashboard")
def teacher_dashboard():
    if session.get("role") != "teacher":
        return "Unauthorized"
    
    auto_delete_expired_visitors()  # 🔴 AUTO DELETE HERE
    conn = sqlite3.connect(LOG_DB)
    c = conn.cursor()

    # Total visitors
    c.execute("SELECT COUNT(*) FROM visitors")
    total_visitors = c.fetchone()[0]

    # Total teachers
    c.execute("SELECT COUNT(*) FROM teachers")
    total_teachers = c.fetchone()[0]

    # Total subjects 🔹 NEW
    c.execute("SELECT COUNT(*) FROM subjects")
    total_subjects = c.fetchone()[0]

    # Attendance per day
    c.execute("""
        SELECT DATE(time_in), COUNT(DISTINCT student_id)
        FROM attendance
        WHERE student_id != 'Unknown'
        GROUP BY DATE(time_in)
        ORDER BY DATE(time_in)
    """)
    rows = c.fetchall()
    conn.close()

    attendance_dates = [r[0] for r in rows]
    attendance_counts = [r[1] for r in rows]

    return render_template(
        "teacher_dashboard.html",
        role=session["role"],   # 🔴 important
        total_faces=len(face_db),          # Students
        total_visitors=total_visitors,     # Visitors
        total_teachers=total_teachers,     # Teachers
        total_subjects=total_subjects,     # 🔹 Subjects
        attendance_dates=json.dumps(attendance_dates),
        attendance_counts=json.dumps(attendance_counts))

# Place this near your dashboard routes
@app.route("/back_to_dashboard")
def back_to_dashboard():
    user_role = session.get("role")  # 'teacher' or 'admin'
    if user_role == "teacher":
        return redirect(url_for("teacher_dashboard"))
    elif user_role == "admin":
        return redirect(url_for("admin_dashboard"))
    else:
        return redirect(url_for("login"))

SYSTEM_STATE_FILE = "system_state.json"

def load_system_id():
    if os.path.exists(SYSTEM_STATE_FILE):
        with open(SYSTEM_STATE_FILE, "r") as f:
            return json.load(f).get("system_id")
    return None

def save_system_id(system_id):
    with open(SYSTEM_STATE_FILE, "w") as f:
        json.dump({"system_id": system_id}, f)

SYSTEM_ID = str(time.time())  # unique per startup
save_system_id(SYSTEM_ID)

@app.before_request
def check_system_validity():
    allowed_routes = ["login", "static"]

    if request.endpoint in allowed_routes or request.endpoint is None:
        return

    if "role" in session:
        if session.get("system_id") != SYSTEM_ID:
            session.clear()
            return redirect("/login")
        
@app.route("/download/logs_csv")
def download_logs_csv():
    conn = sqlite3.connect(LOG_DB)

    query = """
        SELECT
            student_id,
            name,
            course,
            subjects,
            time_in,
            score,
            image
        FROM logs
        ORDER BY time_in DESC
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        return "No logs data available", 204

    # 🔥 Add full URL for images
    df["image_url"] = df["image"].apply(
        lambda x: f"http://127.0.0.1:5000/face_snapshots/{x}" if x else ""
    )

    reports_dir = os.path.join(BASE_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    filename = f"logs_with_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    filepath = os.path.join(reports_dir, filename)

    df.to_csv(filepath, index=False)

    return send_file(filepath, as_attachment=True)

if __name__ == "__main__":
    arduino_thread = threading.Thread(target=read_arduino, daemon=True)
    arduino_thread.start()
    for cam_id, src in CAMERA_SOURCES.items():
        threading.Thread(
            target=rtsp_reader,
            args=(cam_id,src),  # pass cam_id now
            daemon=True
        ).start()
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader= False)