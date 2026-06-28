Attention and Posture Monitor:

The program asses live camera footage or pre-recorded video to track if the user is paying attention, has correct posture and measures how distracted / attentive they are.

CV techniques used:

| Capability | File | Requirement area |
|---|---|---|
| Face detection / face landmarks | `gaze.py`, `head_pose.py`, `blink.py` | Face detection, keypoint detection |
| Gaze direction | `gaze.py` | Keypoint detection |
| Head orientation | `head_pose.py` | Face/keypoint-based pose estimation |
| Blink and drowsiness | `blink.py` | Video processing |
| Posture estimation | `posture.py` | Keypoint detection |
| Distraction object detection | `posture.py` | Object detection using YOLOv8n pretrained |
| Custom attention classification | `custom_model.py` | Model trained on custom data |


Project structure:

Attention-Detector/
/data               - stores recorded training data and samples through web app. Tracks each frame with person who recorded, time_stamp and features from otherm odules.
/models             - stores the model used for the web app.
app.py              - streamlit web application that uses the models and its respective modules to monitor a user's posture and attentiveness.
blink.py            - tracks blinks, eye closures and mouth position to check if the user is dozing off or yawning and scores.
custom_model.py     - 
fusion.py           - 
gaze.py             - tracks eye position and direction and scores.
head_pose.py        - tracks head position, roll, orientation, direction etc. to check if user is looking at their screen and scores
posture.py          - tracks posture, shoulder tilt, neck lean and scores
yolov8n.pt          - pretrained model


Setup:

Install python 3.11 if you don't have it

1) Open the project folder and install a virtual environment for py3.11 (python -m venv .venv)
2) Activate the venv and install packages (pip install -r requirements.txt)
3) run streamlit (stream run app.py)
4) Use the sidebar on the left of the dashboard to enable the camera, train models, etc.


Datasets are stored in ./data
Models are saved in ./models
Metrics are stored in ./models/custom_metrics

Score representation:

Score >= 75 : Focused, Looking at screen, normal posture
Score >= 50 : Moderate, looking slightly away or leaning a bit
Score >= 25 : Distracted, Looking awawy, bad posture, eyes closed
Score >= 0 : Very low