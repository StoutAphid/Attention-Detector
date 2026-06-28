# Attention Detector — CSCI435 Computer Vision Project

Real-time attention scoring using a webcam, uploaded image, or uploaded video. The system combines gaze direction, face/eye detection, head orientation, blink/drowsiness, posture, and YOLO distraction detection.

## What changed for the project requirements

- **Pre-trained model:** YOLOv8n is used for distraction object detection, and MediaPipe pre-trained landmark models are used for face mesh and pose estimation.
- **Custom trained model:** `custom_model.py` lets the group collect labelled samples from each member and train a RandomForest attention classifier on the collected dataset.
- **Two input modalities:** Webcam, image upload, and video upload are supported.
- **Clear visual output:** Face bounding box, eye bounding boxes, head orientation text, blink text, posture text, YOLO object boxes, and final attention overlay are displayed.
- **Head orientation fixed:** Pitch, yaw, and roll are shown in the dashboard and on the frame instead of staying at zero.
- **Eye dot removed:** The iris dot/circle overlay has been removed; eyes are now shown using bounding boxes and labels.

## CV techniques used

| Capability | File | Requirement area |
|---|---|---|
| Face detection / face landmarks | `gaze.py`, `head_pose.py`, `blink.py` | Face detection, keypoint detection |
| Gaze direction | `gaze.py` | Keypoint detection |
| Head orientation | `head_pose.py` | Face/keypoint-based pose estimation |
| Blink and drowsiness | `blink.py` | Video processing |
| Posture estimation | `posture.py` | Keypoint detection |
| Distraction object detection | `posture.py` | Object detection using YOLOv8n pretrained |
| Custom attention classification | `custom_model.py` | Model trained on custom data |

## Project structure

```text
attention-detector/
├── app.py                       # Streamlit app
├── gaze.py                      # Face + eye boxes and gaze direction
├── head_pose.py                 # Pitch, yaw, roll head orientation
├── blink.py                     # EAR/MAR blink and drowsiness
├── posture.py                   # Pose posture + YOLO distractions
├── fusion.py                    # Pre-trained/rule-based weighted score fusion
├── custom_model.py              # Collect dataset + train custom model
├── yolov8n.pt                   # Pre-trained YOLOv8 nano weights
├── requirements.txt
├── data/                        # Created automatically when collecting samples
└── models/                      # Created automatically after training custom model
```

## Setup

Use Python 3.9 to 3.11.

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Run:

```bash
streamlit run app.py
```

## How to use the custom model

1. Open the app.
2. Keep **Model** set to **Pre-trained + rules** while collecting data.
3. In the sidebar, enter the group member name.
4. Select the correct label: `focused`, `moderate`, or `distracted`.
5. Turn on **Auto-save labelled samples**.
6. Collect balanced examples from every group member. Do focused and distracted examples for each person.
7. Click **Train / update custom model**.
8. Change **Model** to **Custom trained model**.

The dataset is saved to:

```text
data/custom_attention_dataset.csv
```

The trained model is saved to:

```text
models/custom_attention_model.pkl
```

Training metrics are saved to:

```text
models/custom_metrics.json
```

## Recommended collection plan

For a clean demo, collect at least:

| Per member | Samples |
|---|---:|
| Focused | 20+ |
| Moderate | 20+ |
| Distracted | 20+ |

Examples:

- **Focused:** looking at the screen, normal posture, no phone.
- **Moderate:** looking slightly away or leaning a bit.
- **Distracted:** looking away, phone visible, bad posture, eyes closed, or yawning.

## Run modes

| Mode | Use |
|---|---|
| Webcam | Live defence demo |
| Upload image | Static test example |
| Upload video | Recorded test examples and performance screenshots |

## Score meaning

| Score | Label |
|---:|---|
| 75-100 | FOCUSED |
| 50-74 | MODERATE |
| 25-49 | DISTRACTED |
| 0-24 | VERY LOW |

## Notes for report

Use screenshots from the app for qualitative results. Use `models/custom_metrics.json` for the custom model accuracy and `FPS` from the dashboard for performance.

## Dependencies

- MediaPipe
- OpenCV
- NumPy
- Streamlit
- Ultralytics YOLOv8
- scikit-learn
- joblib
