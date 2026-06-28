"""
this module detects the head position of the user , including their orientation, angle and direction.
returns a 0-100 attention sub-score.
it is also the first process of the pipeline.
"""

import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# creates a facemesh detector which tracks the head position (refine_landmarks = false means it does not extract info from eyes or lips) for one face. 
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.55,
    min_tracking_confidence=0.55,
)

# landmarks for points in face
NOSE = 1
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
LEFT_MOUTH = 61
RIGHT_MOUTH = 291
CHIN = 152

_history = deque(maxlen=8)


def _pt(landmarks, idx, w, h):
    return np.array(
        [landmarks[idx].x * w, landmarks[idx].y * h],
        dtype=np.float32,
    )


def _estimate_angles(frame, landmarks):
    h, w = frame.shape[:2]

    nose = _pt(landmarks, NOSE, w, h)
    left_eye = _pt(landmarks, LEFT_EYE_OUTER, w, h)
    right_eye = _pt(landmarks, RIGHT_EYE_OUTER, w, h)
    left_mouth = _pt(landmarks, LEFT_MOUTH, w, h)
    right_mouth = _pt(landmarks, RIGHT_MOUTH, w, h)
    chin = _pt(landmarks, CHIN, w, h)

    eye_mid = (left_eye + right_eye) / 2
    mouth_mid = (left_mouth + right_mouth) / 2

    eye_width = np.linalg.norm(right_eye - left_eye) + 1e-6
    face_height = np.linalg.norm(chin - eye_mid) + 1e-6

    # Roll = eye line tilt
    roll = np.degrees(
        np.arctan2(
            right_eye[1] - left_eye[1],
            right_eye[0] - left_eye[0],
        )
    )

    # Yaw = nose shifts left/right compared to eye center
    yaw = ((nose[0] - eye_mid[0]) / eye_width) * 75

    # Pitch = nose moves up/down compared to face height
    neutral_nose_y = eye_mid[1] + face_height * 0.32
    pitch = ((nose[1] - neutral_nose_y) / face_height) * 95

    # Clamp unrealistic values
    pitch = float(np.clip(pitch, -55, 55))
    yaw = float(np.clip(yaw, -55, 55))
    roll = float(np.clip(roll, -45, 45))

    return pitch, yaw, roll, nose, eye_mid, mouth_mid


def _orientation_label(pitch, yaw, roll):
    parts = []

    if yaw > 16:
        parts.append("RIGHT")
    elif yaw < -16:
        parts.append("LEFT")

    if pitch > 14:
        parts.append("DOWN")
    elif pitch < -14:
        parts.append("UP")

    if abs(roll) > 16:
        parts.append("TILTED")

    return "CENTER" if not parts else "+".join(parts)

# score by default is 100 but gets penalized when head is away from center
def _raw_score(pitch, yaw, roll):
    score = 100

    score -= max(0, abs(yaw) - 10) * 1.7
    score -= max(0, abs(pitch) - 10) * 1.6
    score -= max(0, abs(roll) - 12) * 1.3

    return int(np.clip(score, 0, 100))

# arrow for direction
def _draw_arrow(frame, nose, pitch, yaw):
    x, y = int(nose[0]), int(nose[1])

    end_x = int(x + yaw * 2.0)
    end_y = int(y + pitch * 2.0)

    cv2.arrowedLine(
        frame,
        (x, y),
        (end_x, end_y),
        (255, 200, 0),
        2,
        tipLength=0.25,
    )


def get_score(frame):
    """
    Args:
        frame: BGR frame

    Returns:
        score: int
        annotated_frame
        details: dict
    """
    out = frame.copy()
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
        cv2.putText(
            out,
            "Head: NO FACE",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 255),
            2,
        )
        return 35, out, {
            "pitch": 0.0,
            "yaw": 0.0,
            "roll": 0.0,
            "orientation": "NO FACE",
        }

    lm = results.multi_face_landmarks[0].landmark
    pitch, yaw, roll, nose, eye_mid, mouth_mid = _estimate_angles(frame, lm)

    orientation = _orientation_label(pitch, yaw, roll)
    raw = _raw_score(pitch, yaw, roll)

    _history.append(raw)
    score = int(np.mean(_history))

    color = (0, 220, 0) if orientation == "CENTER" else (0, 120, 255)

    _draw_arrow(out, nose, pitch, yaw)

    cv2.putText(
        out,
        f"Head: {orientation} | P:{pitch:.1f} Y:{yaw:.1f} R:{roll:.1f}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        color,
        2,
    )

    return score, out, {
        "pitch": float(pitch),
        "yaw": float(yaw),
        "roll": float(roll),
        "orientation": orientation,
    }