"""
app.py - Attention Detector
Run with: streamlit run app.py

Input modes:
1) Webcam
2) Upload video

Training data collection:
1) Webcam live collection
2) Uploaded video collection

Model modes:
1) Pre-trained + rules
2) Custom trained model
"""

import tempfile
import time
from pathlib import Path

import cv2
import streamlit as st

import blink
import custom_model
import fusion
import gaze
import head_pose
import posture


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Attention Detector",
    page_icon="E",
    layout="wide",
)

st.title("Attention Detector")


# -----------------------------
# Session state
# -----------------------------
def _init_state():
    defaults = {
        "last_posture": (50, None, {"distraction_count": 0, "distractions": []}),
        "last_collect_time": 0.0,
        "collected_this_run": 0,
        "fps": 0.0,
        "last_custom_pred": None,
        "training_now": False,
    }

    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


_init_state()


def _safe_dataset_summary():
    try:
        return custom_model.dataset_summary()
    except Exception:
        return {
            "total": 0,
            "by_label": {},
            "by_member": {},
            "model_exists": False,
            "metrics": {},
        }


def _fuse_pretrained(scores, details):
    """
    Uses improved fusion.py if available.
    Falls back to old fusion.py if your file still only accepts one argument.
    """
    try:
        return fusion.fuse(scores, details)
    except TypeError:
        return fusion.fuse(scores)


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Settings")

    input_mode = st.radio(
        "Input",
        ["Webcam", "Upload video"],
        index=0,
    )

    model_mode = st.radio(
        "Model",
        ["Pre-trained + rules", "Custom trained model"],
        index=0,
    )

    st.divider()
    st.subheader("Processing")

    # Vision modules are always enabled.
    show_gaze = True
    show_pose = True
    show_blink = True
    show_posture = True
    enable_alert = True

    yolo_every = st.slider(
        "Run YOLO every N frames",
        min_value=1,
        max_value=20,
        value=6,
    )

    st.divider()
    st.subheader("Custom model")

    summary = _safe_dataset_summary()

    st.caption(
        f"Dataset: {summary.get('total', 0)} samples - "
        f"Model: {'ready' if summary.get('model_exists') else 'not trained'}"
    )

    if summary.get("by_label"):
        st.write("Label counts:")
        st.json(summary["by_label"], expanded=False)

    with st.expander("How to create the custom model", expanded=summary.get("total", 0) < 12):
        st.markdown(
            """
**Option 1: Webcam**
1. Select **Webcam**.
2. Enter the group member name.
3. Pick the correct label.
4. Turn on **Auto collect webcam samples**.
5. Stay in that label/pose for 10-20 seconds.
6. Repeat for all labels and all group members.

**Option 2: Video upload**
1. Select **Upload video**.
2. Upload a video that mostly matches one label.
3. Turn on **Use this video for custom training data**.
4. Pick the label for that video.
5. Process the video.
6. Repeat for all labels and all group members.

Then click **Train custom model**.
"""
        )

    member_name = st.text_input("Group member name", value="member_1")

    sample_label = st.selectbox(
        "Label for collected samples",
        custom_model.LABELS,
    )

    webcam_collect_on = False
    webcam_collect_every = 12
    webcam_save_frames = False

    if input_mode == "Webcam":
        webcam_collect_on = st.toggle("Auto collect webcam samples", value=False)

        webcam_collect_every = st.slider(
            "Save webcam sample every N frames",
            min_value=3,
            max_value=60,
            value=12,
        )

        webcam_save_frames = st.toggle("Also save webcam sample images", value=False)

        if webcam_collect_on:
            st.warning("COLLECTING WEBCAM DATA ONLY - the model is NOT training yet.")
        else:
            st.info("Webcam data collection is off.")

    else:
        st.info("For video training, use the video upload controls on the main page.")

    st.caption("Collecting samples only saves labelled data. Training only happens when you click the button below.")

    train_status = st.empty()

    if st.button("Train custom model"):
        try:
            st.session_state["training_now"] = True
            train_status.warning("TRAINING MODEL... please wait.")

            with st.spinner("Training custom model..."):
                metrics = custom_model.train_model()

            st.session_state["training_now"] = False

            accuracy = metrics.get("accuracy", "N/A")
            total_samples = metrics.get("total_samples", summary.get("total", "N/A"))

            train_status.success(
                f"MODEL TRAINED | Accuracy: {accuracy} | Samples: {total_samples}"
            )

        except Exception as e:
            st.session_state["training_now"] = False
            train_status.error(str(e))

    st.divider()

    if input_mode == "Webcam":
        run_camera = st.toggle("Start camera", value=False, key="run_camera")
    else:
        run_camera = False

    if st.button("Reset session stats"):
        fusion.reset_session()
        st.session_state["collected_this_run"] = 0
        st.success("Session reset.")


# -----------------------------
# Main layout
# -----------------------------
col1, col2 = st.columns([2.4, 1])

with col1:
    st.subheader("Output")
    feed_placeholder = st.empty()
    video_status_placeholder = st.empty()
    download_placeholder = st.empty()

with col2:
    st.subheader("Attention Score")
    score_placeholder = st.empty()
    label_placeholder = st.empty()
    alert_placeholder = st.empty()
    model_placeholder = st.empty()
    collect_placeholder = st.empty()


# -----------------------------
# Core pipeline
# -----------------------------
def _run_pipeline(
    frame,
    frame_count=1,
    collect_sample=False,
    collect_member=None,
    collect_label=None,
    collect_every=12,
    save_sample_frame=False,
    collect_source="webcam",
):
    start = time.perf_counter()

    # Gaze / face / eyes
    if show_gaze:
        g_score, frame, gaze_info = gaze.get_score(frame)
    else:
        g_score = 50
        gaze_info = {"direction": "OFF"}

    # Head orientation
    if show_pose:
        h_score, frame, head_info = head_pose.get_score(frame)
    else:
        h_score = 50
        head_info = {
            "pitch": 0.0,
            "yaw": 0.0,
            "roll": 0.0,
            "orientation": "OFF",
        }

    # Blink / drowsiness
    if show_blink:
        b_score, frame, blink_info = blink.get_score(frame)
    else:
        b_score = 50
        blink_info = {}

    # Posture + YOLO distractions
    if show_posture:
        run_yolo = frame_count % yolo_every == 0

        if run_yolo or st.session_state["last_posture"][1] is None:
            p_score, frame, posture_info = posture.get_score(
                frame,
                run_yolo=run_yolo,
            )

            st.session_state["last_posture"] = (
                p_score,
                frame.copy(),
                posture_info,
            )

        else:
            _, _, posture_info = st.session_state["last_posture"]

            posture_only_score, frame, posture_info_no_yolo = posture.get_score(
                frame,
                run_yolo=False,
            )

            last_penalty = int(posture_info.get("distraction_penalty", 0))
            p_score = max(0, posture_only_score - last_penalty)

            posture_info["posture"] = posture_info_no_yolo.get(
                "posture",
                posture_only_score,
            )

            posture_info["posture_details"] = posture_info_no_yolo.get(
                "posture_details",
                {},
            )

    else:
        p_score = 50
        posture_info = {
            "distraction_count": 0,
            "distractions": [],
        }

    scores = {
        "gaze": g_score,
        "head": h_score,
        "blink": b_score,
        "posture": p_score,
    }

    pretrained_details = {
        "gaze": gaze_info,
        "head": head_info,
        "blink": blink_info,
        "posture": posture_info,
    }

    pretrained_score = _fuse_pretrained(scores, pretrained_details)

    features = custom_model.build_features(
        scores,
        gaze_info,
        head_info,
        blink_info,
        posture_info,
    )

    custom_pred = None
    custom_probs = {}

    if model_mode == "Custom trained model":
        try:
            final_score, custom_pred, custom_probs = custom_model.predict_score(features)

        except Exception as e:
            final_score = pretrained_score
            custom_pred = f"not ready: {e}"

    else:
        final_score = pretrained_score

    st.session_state["last_custom_pred"] = custom_pred

    label, color = fusion.get_label(final_score)

    score_color = (
        (0, 200, 0)
        if final_score >= 75
        else (0, 165, 255)
        if final_score >= 50
        else (0, 0, 255)
    )

    cv2.putText(
        frame,
        f"ATTENTION: {final_score} - {label}",
        (10, frame.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        score_color,
        2,
    )

    # Data collection only, not training
    if collect_sample and frame_count % collect_every == 0:
        now = time.time()

        if now - st.session_state["last_collect_time"] > 0.15:
            member_to_save = collect_member or member_name
            label_to_save = collect_label or sample_label

            custom_model.append_sample(
                member_to_save,
                label_to_save,
                features,
                frame=frame,
                save_frame=save_sample_frame,
            )

            st.session_state["last_collect_time"] = now
            st.session_state["collected_this_run"] += 1

    elapsed = max(time.perf_counter() - start, 1e-6)
    st.session_state["fps"] = round(1.0 / elapsed, 1)

    return frame, final_score, label, color, custom_pred, custom_probs


def _update_dashboard(
    frame,
    final_score,
    label,
    color,
    custom_pred=None,
    custom_probs=None,
    collecting=False,
    collecting_label=None,
    collecting_source=None,
):
    feed_placeholder.image(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        channels="RGB",
        use_column_width=True,
    )

    score_placeholder.markdown(
        f"""
        <h1 style='text-align:center; color:{color}; font-size:84px; margin-bottom:0'>
            {final_score}
        </h1>
        """,
        unsafe_allow_html=True,
    )

    label_placeholder.markdown(
        f"<h3 style='text-align:center; color:{color}; margin-top:0'>{label}</h3>",
        unsafe_allow_html=True,
    )

    if enable_alert and fusion.should_alert(final_score):
        alert_placeholder.error("⚠️ Attention dropped - refocus!")
    else:
        alert_placeholder.empty()

    if model_mode == "Custom trained model":
        model_text = f"Custom model: {custom_pred or 'running'}"
    else:
        model_text = "Pre-trained + rules"

    model_placeholder.info(
        f"{model_text} | FPS: {st.session_state['fps']}"
    )

    if st.session_state.get("training_now"):
        collect_placeholder.warning("TRAINING MODEL...")

    elif collecting:
        collect_placeholder.warning(
            f"COLLECTING {str(collecting_source).upper()} DATA ONLY - "
            f"Label: {collecting_label} | "
            f"Saved this run: {st.session_state['collected_this_run']}"
        )

    else:
        collect_placeholder.info(
            f"Not collecting | Saved this run: {st.session_state['collected_this_run']}"
        )


# -----------------------------
# Uploaded video processing
# -----------------------------
def _process_uploaded_video(
    uploaded_file,
    frame_stride,
    max_frames,
    preview_every,
    save_output,
    collect_video_samples,
    video_member,
    video_label,
    video_collect_every,
    video_save_frames,
):
    suffix = Path(uploaded_file.name).suffix or ".mp4"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        input_path = tmp.name

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        st.error("Could not read uploaded video.")
        return

    total_source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    writer = None
    output_path = None

    if save_output and width > 0 and height > 0:
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

        output_fps = max(1.0, source_fps / max(1, frame_stride))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            output_fps,
            (width, height),
        )

        if not writer.isOpened():
            writer = None
            output_path = None
            st.warning("Could not create output video file. Live processing will still work.")

    progress = st.progress(0, text="Processing video...")

    source_frame_count = 0
    processed_count = 0
    last_result = None

    while processed_count < max_frames:
        ret, frame = cap.read()

        if not ret:
            break

        source_frame_count += 1

        if source_frame_count % frame_stride != 0:
            continue

        processed_count += 1

        result = _run_pipeline(
            frame,
            frame_count=processed_count,
            collect_sample=collect_video_samples,
            collect_member=video_member,
            collect_label=video_label,
            collect_every=video_collect_every,
            save_sample_frame=video_save_frames,
            collect_source="video",
        )

        annotated_frame = result[0]
        last_result = result

        if writer is not None:
            if annotated_frame.shape[1] != width or annotated_frame.shape[0] != height:
                annotated_frame = cv2.resize(annotated_frame, (width, height))

            writer.write(annotated_frame)

        if processed_count == 1 or processed_count % preview_every == 0:
            _update_dashboard(
                *result,
                collecting=collect_video_samples,
                collecting_label=video_label,
                collecting_source="video",
            )

        if total_source_frames > 0:
            progress_value = min(source_frame_count / total_source_frames, 1.0)
        else:
            progress_value = min(processed_count / max_frames, 1.0)

        progress.progress(
            progress_value,
            text=f"Processed {processed_count} frames | Saved samples: {st.session_state['collected_this_run']}",
        )

    cap.release()

    if writer is not None:
        writer.release()

    progress.empty()

    if last_result is not None:
        _update_dashboard(
            *last_result,
            collecting=collect_video_samples,
            collecting_label=video_label,
            collecting_source="video",
        )

        if collect_video_samples:
            video_status_placeholder.success(
                f"Done. Processed {processed_count} frames and collected video training samples."
            )
        else:
            video_status_placeholder.success(
                f"Done. Processed {processed_count} frames."
            )

    else:
        video_status_placeholder.warning("No frames were processed.")

    if output_path and Path(output_path).exists():
        with open(output_path, "rb") as f:
            video_bytes = f.read()

        download_placeholder.download_button(
            "Download annotated video",
            data=video_bytes,
            file_name="annotated_attention_output.mp4",
            mime="video/mp4",
        )


# -----------------------------
# Input modes
# -----------------------------
if input_mode == "Upload video":
    uploaded = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"],
    )

    st.subheader("Video processing settings")

    settings_col1, settings_col2, settings_col3 = st.columns(3)

    with settings_col1:
        frame_stride = st.number_input(
            "Process every N source frames",
            min_value=1,
            max_value=30,
            value=5,
            step=1,
        )

    with settings_col2:
        max_frames = st.number_input(
            "Max processed frames",
            min_value=30,
            max_value=5000,
            value=500,
            step=50,
        )

    with settings_col3:
        preview_every = st.number_input(
            "Preview every N processed frames",
            min_value=1,
            max_value=100,
            value=10,
            step=1,
        )

    save_output_video = st.checkbox(
        "Create downloadable annotated video",
        value=True,
    )

    st.divider()
    st.subheader("Use uploaded video for custom training")

    collect_video_samples = st.checkbox(
        "Use this video for custom training data",
        value=False,
    )

    video_member = st.text_input(
        "Video group member name",
        value=member_name,
    )

    video_label = st.selectbox(
        "Label for this whole video",
        custom_model.LABELS,
        key="video_label_select",
    )

    video_collect_every = st.slider(
        "Save video sample every N processed frames",
        min_value=1,
        max_value=60,
        value=5,
    )

    video_save_frames = st.checkbox(
        "Also save video sample images",
        value=False,
    )

    if collect_video_samples:
        st.warning(
            "VIDEO DATA COLLECTION IS ON. Make sure this whole video mostly matches the selected label."
        )
    else:
        st.info("Video will be processed only. No training samples will be saved.")

    start_video = st.button("Process video")

    if uploaded and start_video:
        download_placeholder.empty()
        video_status_placeholder.empty()

        _process_uploaded_video(
            uploaded_file=uploaded,
            frame_stride=int(frame_stride),
            max_frames=int(max_frames),
            preview_every=int(preview_every),
            save_output=bool(save_output_video),
            collect_video_samples=bool(collect_video_samples),
            video_member=video_member,
            video_label=video_label,
            video_collect_every=int(video_collect_every),
            video_save_frames=bool(video_save_frames),
        )

    elif not uploaded:
        feed_placeholder.info("Upload a video to process it.")


else:
    if not run_camera:
        feed_placeholder.info("Toggle **Start camera** in the sidebar to begin.")

    else:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("❌ Could not open webcam. Make sure it is connected and not used by another app.")
            st.stop()

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        frame_count = 0

        while run_camera:
            ok, frame = cap.read()

            if not ok:
                st.error("❌ Lost camera feed.")
                break

            frame_count += 1

            result = _run_pipeline(
                frame,
                frame_count=frame_count,
                collect_sample=webcam_collect_on,
                collect_member=member_name,
                collect_label=sample_label,
                collect_every=webcam_collect_every,
                save_sample_frame=webcam_save_frames,
                collect_source="webcam",
            )

            _update_dashboard(
                *result,
                collecting=webcam_collect_on,
                collecting_label=sample_label,
                collecting_source="webcam",
            )

            time.sleep(0.01)

            run_camera = st.session_state.get("run_camera", True)

        cap.release()