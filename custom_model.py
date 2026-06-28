"""
custom_model.py — Custom attention model support

This module lets the group collect labelled feature rows from each member and
train a custom classifier on that dataset. It does not replace the CV pipeline;
it learns from the outputs of gaze, head pose, blink, posture, and distraction
modules.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Dict, Tuple, Any

import joblib
import numpy as np

DATA_DIR = Path("data")
SAMPLE_DIR = DATA_DIR / "samples"
MODEL_DIR = Path("models")
DATASET_PATH = DATA_DIR / "custom_attention_dataset.csv"
MODEL_PATH = MODEL_DIR / "custom_attention_model.pkl"
METRICS_PATH = MODEL_DIR / "custom_metrics.json"

LABELS = ["focused", "moderate", "distracted"]

FEATURE_COLUMNS = [
    "gaze_score",
    "gaze_off_center",
    "head_score",
    "abs_pitch",
    "abs_yaw",
    "abs_roll",
    "blink_score",
    "ear",
    "mar",
    "bpm",
    "posture_score",
    "distraction_count",
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def build_features(
    scores: Dict[str, int],
    gaze_info: Any,
    head_info: Any,
    blink_info: Any,
    posture_info: Any,
) -> Dict[str, float]:
    """Create one numeric feature row from the current CV outputs."""
    if not isinstance(gaze_info, dict):
        gaze_info = {"direction": gaze_info}
    if not isinstance(head_info, dict):
        if isinstance(head_info, (tuple, list)) and len(head_info) == 3:
            head_info = {"pitch": head_info[0], "yaw": head_info[1], "roll": head_info[2]}
        else:
            head_info = {}
    if not isinstance(blink_info, dict):
        blink_info = {}
    if not isinstance(posture_info, dict):
        posture_info = {}

    direction = str(gaze_info.get("direction", "NO FACE")).upper()
    return {
        "gaze_score": _safe_float(scores.get("gaze"), 50),
        "gaze_off_center": 0.0 if direction == "CENTER" else 1.0,
        "head_score": _safe_float(scores.get("head"), 50),
        "abs_pitch": abs(_safe_float(head_info.get("pitch"))),
        "abs_yaw": abs(_safe_float(head_info.get("yaw"))),
        "abs_roll": abs(_safe_float(head_info.get("roll"))),
        "blink_score": _safe_float(scores.get("blink"), 50),
        "ear": _safe_float(blink_info.get("ear"), 0.0),
        "mar": _safe_float(blink_info.get("mar"), 0.0),
        "bpm": _safe_float(blink_info.get("bpm"), 0.0),
        "posture_score": _safe_float(scores.get("posture"), 50),
        "distraction_count": _safe_float(posture_info.get("distraction_count"), 0),
    }


def append_sample(
    member: str,
    label: str,
    features: Dict[str, float],
    frame=None,
    save_frame: bool = True,
) -> str:
    """Append one labelled training sample to CSV. Optionally saves the frame."""
    if label not in LABELS:
        raise ValueError(f"Label must be one of: {', '.join(LABELS)}")

    DATA_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    member_clean = (member or "unknown").strip().replace(" ", "_") or "unknown"
    timestamp = int(time.time() * 1000)
    image_path = ""

    if save_frame and frame is not None:
        import cv2

        out_dir = SAMPLE_DIR / member_clean / label
        out_dir.mkdir(parents=True, exist_ok=True)
        image_path = str(out_dir / f"{timestamp}.jpg")
        cv2.imwrite(image_path, frame)

    row = {
        "timestamp": timestamp,
        "member": member_clean,
        "label": label,
        "image_path": image_path,
    }
    row.update({col: float(features.get(col, 0.0)) for col in FEATURE_COLUMNS})

    file_exists = DATASET_PATH.exists()
    with DATASET_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["timestamp", "member", "label", "image_path"] + FEATURE_COLUMNS,
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return str(DATASET_PATH)


def _read_dataset() -> Tuple[np.ndarray, np.ndarray, list]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError("No custom dataset found yet. Collect samples first.")

    rows = []
    with DATASET_PATH.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError("Dataset exists but contains no rows.")

    X = np.array(
        [[_safe_float(row.get(col), 0.0) for col in FEATURE_COLUMNS] for row in rows],
        dtype=np.float32,
    )
    y = np.array([row.get("label", "") for row in rows])
    return X, y, rows


def dataset_summary() -> Dict[str, Any]:
    """Return sample counts by member and label."""
    if not DATASET_PATH.exists():
        return {"total": 0, "by_label": {}, "by_member": {}, "model_exists": MODEL_PATH.exists()}

    _, y, rows = _read_dataset()
    by_label = {label: int(np.sum(y == label)) for label in sorted(set(y))}
    by_member: Dict[str, int] = {}
    for row in rows:
        member = row.get("member", "unknown") or "unknown"
        by_member[member] = by_member.get(member, 0) + 1

    metrics = {}
    if METRICS_PATH.exists():
        try:
            metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        except Exception:
            metrics = {}

    return {
        "total": len(rows),
        "by_label": by_label,
        "by_member": by_member,
        "model_exists": MODEL_PATH.exists(),
        "metrics": metrics,
    }


def train_model() -> Dict[str, Any]:
    """Train/update the custom model from the collected CSV dataset."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split

    X, y, rows = _read_dataset()
    unique_labels = sorted(set(y))
    if len(unique_labels) < 2:
        raise ValueError("Need at least 2 labels before training, e.g. focused and distracted.")
    if len(rows) < 12:
        raise ValueError("Collect at least 12 total samples before training.")

    label_counts = {label: int(np.sum(y == label)) for label in unique_labels}
    can_stratify = min(label_counts.values()) >= 2 and len(rows) >= len(unique_labels) * 2

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y if can_stratify else None,
    )

    clf = RandomForestClassifier(
        n_estimators=160,
        max_depth=8,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred)) if len(y_test) else 0.0

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": clf, "features": FEATURE_COLUMNS}, MODEL_PATH)

    metrics = {
        "accuracy": round(accuracy, 3),
        "total_samples": len(rows),
        "label_counts": label_counts,
        "features": FEATURE_COLUMNS,
        "trained_at": int(time.time()),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def load_model():
    if not MODEL_PATH.exists():
        return None
    try:
        bundle = joblib.load(MODEL_PATH)
        return bundle.get("model") if isinstance(bundle, dict) else bundle
    except Exception:
        return None


def predict_score(features: Dict[str, float]) -> Tuple[int, str, Dict[str, float]]:
    """Predict final attention score using the trained custom model."""
    model = load_model()
    if model is None:
        raise FileNotFoundError("Custom model not found. Train it first.")

    X = np.array([[float(features.get(col, 0.0)) for col in FEATURE_COLUMNS]], dtype=np.float32)
    pred = str(model.predict(X)[0])

    probs: Dict[str, float] = {}
    if hasattr(model, "predict_proba"):
        raw = model.predict_proba(X)[0]
        probs = {str(label): float(prob) for label, prob in zip(model.classes_, raw)}

    # Convert class probabilities to a 0-100 attention score.
    if probs:
        score = (
            probs.get("focused", 0.0) * 100
            + probs.get("moderate", 0.0) * 60
            + probs.get("distracted", 0.0) * 15
        )
    else:
        score = {"focused": 100, "moderate": 60, "distracted": 15}.get(pred, 50)

    return int(np.clip(score, 0, 100)), pred, probs
