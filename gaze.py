"""
gaze.py — Gaze direction tracking
Detects iris position relative to eye bounding boxes.
Returns a 0-100 attention sub-score.
"""

import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,  # required for iris landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

_history = []
HISTORY_LEN = 10


def _points(landmarks, idxs, w, h):
    return np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in idxs], dtype=np.float32)


def _bbox_from_points(pts, pad=4, w=None, h=None):
    x1, y1 = pts[:, 0].min() - pad, pts[:, 1].min() - pad
    x2, y2 = pts[:, 0].max() + pad, pts[:, 1].max() + pad
    if w is not None and h is not None:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
    return int(x1), int(y1), int(x2), int(y2)


def _draw_box_label(frame, box, label, color):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text_y = max(18, y1 - 8)
    cv2.putText(frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def _iris_ratio(landmarks, iris_idx, eye_idx, w, h):
    iris_pts = _points(landmarks, iris_idx, w, h)
    eye_pts = _points(landmarks, eye_idx, w, h)
    cx, cy = iris_pts.mean(axis=0)
    x_min, x_max = eye_pts[:, 0].min(), eye_pts[:, 0].max()
    y_min, y_max = eye_pts[:, 1].min(), eye_pts[:, 1].max()
    rx = (cx - x_min) / (x_max - x_min + 1e-6)
    ry = (cy - y_min) / (y_max - y_min + 1e-6)
    return float(rx), float(ry)


def _classify(rx, ry):
    if rx < 0.38:
        return "LEFT"
    if rx > 0.62:
        return "RIGHT"
    if ry < 0.38:
        return "UP"
    if ry > 0.65:
        return "DOWN"
    return "CENTER"


def _score(direction):
    return {"CENTER": 100, "UP": 70, "DOWN": 40, "LEFT": 30, "RIGHT": 30}.get(direction, 50)


def get_score(frame):
    """
    Args:   BGR frame
    Returns: (score: int 0-100, annotated_frame, details: dict)
    """
    global _history
    h, w = frame.shape[:2]
    out = frame.copy()
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
        cv2.putText(out, "Face: NOT DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        return 50, out, {"direction": "NO FACE"}

    lm = results.multi_face_landmarks[0].landmark
    all_pts = _points(lm, range(len(lm)), w, h)
    left_eye_pts = _points(lm, LEFT_EYE, w, h)
    right_eye_pts = _points(lm, RIGHT_EYE, w, h)

    face_box = _bbox_from_points(all_pts, pad=8, w=w, h=h)
    left_eye_box = _bbox_from_points(left_eye_pts, pad=6, w=w, h=h)
    right_eye_box = _bbox_from_points(right_eye_pts, pad=6, w=w, h=h)

    lx, ly = _iris_ratio(lm, LEFT_IRIS, LEFT_EYE, w, h)
    rx, ry = _iris_ratio(lm, RIGHT_IRIS, RIGHT_EYE, w, h)
    avg_rx, avg_ry = (lx + rx) / 2, (ly + ry) / 2
    direction = _classify(avg_rx, avg_ry)

    _history.append(_score(direction))
    if len(_history) > HISTORY_LEN:
        _history.pop(0)
    score = int(np.mean(_history))

    color = (0, 220, 0) if direction == "CENTER" else (0, 120, 255)
    _draw_box_label(out, face_box, "FACE", (0, 220, 0))
    _draw_box_label(out, left_eye_box, "LEFT EYE", color)
    _draw_box_label(out, right_eye_box, "RIGHT EYE", color)

    # Intentionally no iris dot/circle. The output now uses boxes + text overlays.
    cv2.putText(out, f"Gaze: {direction} | Score: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)

    details = {
        "direction": direction,
        "left_ratio": (lx, ly),
        "right_ratio": (rx, ry),
        "avg_ratio": (avg_rx, avg_ry),
        "face_box": face_box,
        "left_eye_box": left_eye_box,
        "right_eye_box": right_eye_box,
    }
    return score, out, details
