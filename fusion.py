"""
fusion.py — Improved pre-trained/rule-based attention fusion

This is the pre-trained mode's decision layer.
It combines gaze, head pose, blink/drowsiness, and posture/distraction scores.
"""

import time
from collections import deque

import numpy as np

WEIGHTS = {
    "gaze": 0.34,
    "head": 0.30,
    "blink": 0.18,
    "posture": 0.18,
}

_session_log = deque(maxlen=7200)
_score_smooth = deque(maxlen=6)

_last_alert = 0
ALERT_COOLDOWN = 8


def _clip(value, low=0, high=100):
    return int(np.clip(value, low, high))


def _penalty_from_details(scores, details):
    penalty = 0

    gaze_info = details.get("gaze", {}) if details else {}
    head_info = details.get("head", {}) if details else {}
    blink_info = details.get("blink", {}) if details else {}
    posture_info = details.get("posture", {}) if details else {}

    gaze_direction = str(gaze_info.get("direction", "")).upper()
    head_orientation = str(head_info.get("orientation", "")).upper()

    if gaze_direction == "NO FACE" or head_orientation == "NO FACE":
        penalty += 25

    if gaze_direction in ["LEFT", "RIGHT"]:
        penalty += 14
    elif gaze_direction == "DOWN":
        penalty += 18
    elif gaze_direction == "UP":
        penalty += 8

    if head_orientation != "CENTER" and head_orientation != "NO FACE":
        penalty += 10

    if bool(blink_info.get("eye_closed", False)):
        penalty += 18

    if float(blink_info.get("mar", 0) or 0) > 0.60:
        penalty += 10

    distractions = int(posture_info.get("distraction_count", 0) or 0)
    if distractions > 0:
        penalty += min(30, distractions * 15)

    # If several modules are weak at the same time, confidence should drop harder.
    low_signals = sum(1 for value in scores.values() if int(value) < 45)
    if low_signals >= 2:
        penalty += 12
    if low_signals >= 3:
        penalty += 15

    return penalty


def fuse(scores: dict, details: dict | None = None) -> int:
    """
    Args:
        scores: {"gaze": int, "head": int, "blink": int, "posture": int}
        details: optional module details from gaze/head/blink/posture

    Returns:
        final attention score from 0-100
    """
    weighted = 0.0

    for key, weight in WEIGHTS.items():
        weighted += int(scores.get(key, 50)) * weight

    penalty = _penalty_from_details(scores, details or {})
    raw_score = _clip(weighted - penalty)

    # Smooth the final score so it does not jump around every frame.
    _score_smooth.append(raw_score)
    smoothed_score = int(np.mean(_score_smooth))

    _session_log.append((time.time(), smoothed_score))
    return smoothed_score


def should_alert(score: int) -> bool:
    global _last_alert

    now = time.time()

    if score < 40 and (now - _last_alert) > ALERT_COOLDOWN:
        _last_alert = now
        return True

    return False


def get_label(score: int) -> tuple:
    if score >= 75:
        return "FOCUSED", "#22c55e"
    if score >= 50:
        return "MODERATE", "#f59e0b"
    if score >= 25:
        return "DISTRACTED", "#ef4444"
    return "VERY LOW", "#7f1d1d"


def session_summary() -> dict:
    if not _session_log:
        return {}

    scores = [s for _, s in _session_log]
    times = [t for t, _ in _session_log]
    duration = times[-1] - times[0] if len(times) > 1 else 0

    focused_pct = sum(1 for s in scores if s >= 75) / len(scores) * 100
    avg = int(np.mean(scores))
    low_periods = sum(1 for s in scores if s < 40)

    return {
        "avg_score": avg,
        "focused_pct": round(focused_pct, 1),
        "low_periods": low_periods,
        "duration_secs": int(duration),
        "history": list(zip(times, scores)),
    }


def reset_session():
    global _last_alert

    _session_log.clear()
    _score_smooth.clear()
    _last_alert = 0