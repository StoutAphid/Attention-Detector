"""
posture.py — Upper body posture + distraction object detection
Uses MediaPipe Pose for shoulder/neck posture.
Uses YOLOv8 pretrained for distracting objects.
Returns a 0-100 attention sub-score.
"""

import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

_yolo_model = None

DISTRACTION_LABELS = {
    "cell phone",
    "remote",
    "book",
    "laptop",
    "keyboard",
    "mouse",
    "tv",
}

_history = []
HISTORY_LEN = 10


def _load_yolo():
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO

            _yolo_model = YOLO("yolov8n.pt")
        except Exception as e:
            print(f"[posture] YOLO not available: {e}")
    return _yolo_model


def _visibility_ok(*pts, threshold=0.45):
    return all(getattr(p, "visibility", 1.0) >= threshold for p in pts)


def _posture_score(landmarks, w, h):
    lm = landmarks
    ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
    rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    le = lm[mp_pose.PoseLandmark.LEFT_EAR]
    re = lm[mp_pose.PoseLandmark.RIGHT_EAR]

    if not _visibility_ok(ls, rs, le, re):
        return 50, {"reason": "LOW VISIBILITY"}

    shoulder_diff_px = abs(ls.y - rs.y) * h
    shoulder_width_px = max(abs(ls.x - rs.x) * w, 1)
    shoulder_tilt = shoulder_diff_px / shoulder_width_px

    ear_mid_x = (le.x + re.x) / 2
    ear_mid_y = (le.y + re.y) / 2
    sh_mid_x = (ls.x + rs.x) / 2
    sh_mid_y = (ls.y + rs.y) / 2

    neck_angle = np.degrees(
        np.arctan2(abs(ear_mid_x - sh_mid_x) * w, abs(sh_mid_y - ear_mid_y) * h + 1e-6)
    )

    score = 100
    if shoulder_diff_px > 20 or shoulder_tilt > 0.08:
        score -= 20
    if neck_angle > 15:
        score -= 30
    if neck_angle > 30:
        score -= 20

    details = {
        "shoulder_diff_px": float(shoulder_diff_px),
        "shoulder_tilt": float(shoulder_tilt),
        "neck_angle": float(neck_angle),
        "reason": "OK" if score >= 70 else "LEANING/SLOUCHING",
    }
    return max(0, score), details


def _draw_upper_body(frame, landmarks, w, h):
    ids = [
        mp_pose.PoseLandmark.LEFT_EAR,
        mp_pose.PoseLandmark.RIGHT_EAR,
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
    ]
    pts = {}
    for idx in ids:
        lm = landmarks[idx]
        pts[idx] = (int(lm.x * w), int(lm.y * h))
        cv2.circle(frame, pts[idx], 4, (255, 200, 0), -1)

    cv2.line(frame, pts[mp_pose.PoseLandmark.LEFT_SHOULDER], pts[mp_pose.PoseLandmark.RIGHT_SHOULDER], (255, 200, 0), 2)
    cv2.line(frame, pts[mp_pose.PoseLandmark.LEFT_EAR], pts[mp_pose.PoseLandmark.RIGHT_EAR], (255, 200, 0), 2)


def _detect_distractions(frame):
    model = _load_yolo()
    if model is None:
        return 0, frame, []

    out = frame.copy()
    found = []
    penalty = 0

    results = model(frame, verbose=False)[0]
    for box in results.boxes:
        label = model.names[int(box.cls)]
        conf = float(box.conf)
        if label in DISTRACTION_LABELS and conf > 0.40:
            found.append({"label": label, "confidence": conf})
            penalty += 45
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                out,
                f"DISTRACTION: {label} {conf:.0%}",
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
            )

    return min(penalty, 100), out, found


def get_score(frame, run_yolo=True):
    """
    Args:   BGR frame
    Returns: (score: int 0-100, annotated_frame, details: dict)
    """
    global _history
    h, w = frame.shape[:2]
    out = frame.copy()

    pose_score = 50
    posture_details = {"reason": "NO POSE"}
    pose_results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if pose_results.pose_landmarks:
        pose_score, posture_details = _posture_score(pose_results.pose_landmarks.landmark, w, h)
        _draw_upper_body(out, pose_results.pose_landmarks.landmark, w, h)

    dist_penalty, out, distractions = _detect_distractions(out) if run_yolo else (0, out, [])
    raw_score = max(0, pose_score - dist_penalty)

    _history.append(raw_score)
    if len(_history) > HISTORY_LEN:
        _history.pop(0)
    score = int(np.mean(_history))

    color = (0, 220, 0) if score > 60 else (0, 120, 255)
    cv2.putText(out, f"Posture: {posture_details.get('reason', 'OK')} | Score {score}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    return score, out, {
        "posture": int(pose_score),
        "posture_details": posture_details,
        "distractions": distractions,
        "distraction_count": len(distractions),
        "distraction_penalty": int(dist_penalty),
    }
