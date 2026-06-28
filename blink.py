"""
blink.py — Blink rate and drowsiness detection
Uses Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR).
Returns a 0-100 attention sub-score.
"""

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

LEFT_EYE_EAR = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]
MOUTH = [13, 14, 78, 308, 82, 87, 312, 317]

EAR_THRESHOLD = 0.21
MAR_THRESHOLD = 0.60
BLINK_CONSEC = 2

_ear_counter = 0
_blink_count = 0
_blink_times = deque(maxlen=30)
_yawn_active = False
_yawn_count = 0
_start_time = time.time()
_history = []
HISTORY_LEN = 10


def _ear(landmarks, eye_idx, w, h):
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in eye_idx])
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C + 1e-6)


def _mar(landmarks, w, h):
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in MOUTH])
    A = np.linalg.norm(pts[0] - pts[1])
    B = np.linalg.norm(pts[2] - pts[3])
    return A / (B + 1e-6)


def _score(avg_ear, mar, blinks_per_min):
    score = 100
    if avg_ear < EAR_THRESHOLD:
        score -= 40
    if mar > MAR_THRESHOLD:
        score -= 25
    if blinks_per_min > 30:
        score -= 20
    elif 0 < blinks_per_min < 5:
        score -= 10
    return max(0, score)


def get_score(frame):
    """
    Args:   BGR frame
    Returns: (score: int 0-100, annotated_frame, details: dict)
    """
    global _ear_counter, _blink_count, _yawn_active, _yawn_count, _history

    h, w = frame.shape[:2]
    out = frame.copy()
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
        cv2.putText(out, "Blink: NO FACE", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 255), 2)
        return 50, out, {}

    lm = results.multi_face_landmarks[0].landmark
    left_ear = _ear(lm, LEFT_EYE_EAR, w, h)
    right_ear = _ear(lm, RIGHT_EYE_EAR, w, h)
    avg_ear = (left_ear + right_ear) / 2
    mar = _mar(lm, w, h)

    eye_closed = avg_ear < EAR_THRESHOLD
    if eye_closed:
        _ear_counter += 1
    else:
        if _ear_counter >= BLINK_CONSEC:
            _blink_count += 1
            _blink_times.append(time.time())
        _ear_counter = 0

    if mar > MAR_THRESHOLD:
        if not _yawn_active:
            _yawn_active = True
            _yawn_count += 1
    else:
        _yawn_active = False

    now = time.time()
    recent_blinks = [t for t in _blink_times if now - t <= 60]
    elapsed = max(now - _start_time, 1)
    bpm = len(recent_blinks) / min(elapsed, 60) * 60

    raw_score = _score(avg_ear, mar, bpm)
    _history.append(raw_score)
    if len(_history) > HISTORY_LEN:
        _history.pop(0)
    score = int(np.mean(_history))

    color = (0, 220, 0) if score > 60 else (0, 120, 255)
    cv2.putText(out, f"Blink: EAR {avg_ear:.2f} | BPM {bpm:.0f} | Score {score}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    if eye_closed:
        cv2.putText(out, "EYES CLOSED", (10, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    if mar > MAR_THRESHOLD:
        cv2.putText(out, "YAWNING", (10, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    return score, out, {
        "ear": float(avg_ear),
        "mar": float(mar),
        "bpm": float(bpm),
        "blink_count": int(_blink_count),
        "yawn_count": int(_yawn_count),
        "eye_closed": bool(eye_closed),
    }
