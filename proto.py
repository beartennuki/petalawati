# vehicle_counter_app_fixed.py
# -----------------------------------------------------------------------------
# Streamlit demo – Vehicle counting with a configurable centre‑line
# Current features:
#   • Bounding‑box drift fixed (only draw tracks updated in current frame)
#   • Auto‑deletes temporary upload after processing
#   • Live counter under flow‑direction selector
#   • Start and Stop buttons (⏹ stops cleanly)
#   • When the user presses ⏹ Stop, the *final* counts are shown
#     immediately in the counter box as well as in the usual summary section.
#   • BUGFIX: UI state now correctly resets after processing stops.
# -----------------------------------------------------------------------------
# Requirements: streamlit, torch, torchvision, opencv‑python, deep‑sort‑realtime
# Tested with Python 3.10 / Streamlit 1.34 / Torch 2.3

import streamlit as st
import torch
import cv2
from pathlib import Path
from collections import defaultdict
from deep_sort_realtime.deepsort_tracker import DeepSort

# -----------------------------------------------------------------------------
# 🚀 Streamlit UI & helpers
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Vehicle Counter")
st.title("🚗 Vehicle Counter")

# ── Session flags ───────────────────────────────────────────────────────────
# Initialize session state variables to track the app's status.
if "stop_flag" not in st.session_state:
    st.session_state.stop_flag = False
if "processing" not in st.session_state:
    st.session_state.processing = False  # This controls the enabled/disabled state of the buttons.

video_file = st.file_uploader(
    "Upload a video (MP4/AVI) or leave blank to use webcam",
    type=["mp4", "avi"],
)
conf_thres = st.slider("YOLO confidence threshold", 0.1, 1.0, 0.4, 0.05)

flow_direction = st.selectbox(
    "Select Flow Direction",
    ("Left to Right", "Right to Left", "Up to Down", "Down to Up"),
)

# Live counter placeholder ----------------------------------------------------
counter_slot = st.empty()

# Control buttons -------------------------------------------------------------
cols = st.columns([1, 1])
start_button = cols[0].button("▶️ Start", type="primary", disabled=st.session_state.processing)
stop_button = cols[1].button("⏹️ Stop", disabled=not st.session_state.processing)

# If the stop button is clicked, set the stop_flag in the session state.
# The running loop will detect this and break.
if stop_button:
    st.session_state.stop_flag = True


# -----------------------------------------------------------------------------
# 📦 Load models (cached)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_models(user_conf):
    """Loads YOLOv5 and DeepSort models from cache."""
    with st.spinner("Loading models..."):
        model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
        model.conf = user_conf
        model.classes = [2, 3, 5, 7]  # COCO vehicle classes: car, motorcycle, bus, truck

        tracker = DeepSort(
            max_age=10,
            n_init=3,
            max_iou_distance=0.7,
            nms_max_overlap=1.0,
            embedder="mobilenet",
            half=True,
        )
    return model, tracker


# Utility to render counts in a given container --------------------------------
def render_counts(container, counts, title):
    """Renders the current vehicle counts in a specified Streamlit container."""
    with container.container():
        st.subheader(title)
        if counts:
            for cls, n in sorted(counts.items()):  # Sort for consistent order
                st.write(f"**{cls}** : {n}")
        else:
            st.write("_No crossings detected._")


# -----------------------------------------------------------------------------
# 🎞️ Main processing loop
# -----------------------------------------------------------------------------
# This block is only entered when the user clicks the "Start" button.
if start_button:
    # Reset session state for a new run.
    st.session_state.stop_flag = False
    st.session_state.processing = True

    # Rerun the script immediately to update the button states.
    # This disables the "Start" button and enables the "Stop" button.
    st.rerun()

# The main processing logic will run only when the `processing` flag is True.
if st.session_state.get("processing", False):
    model, tracker = load_models(conf_thres)

    # --- Video capture setup ---
    temp_path: Path | None = None
    if video_file is None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot open webcam.")
            st.session_state.processing = False
            st.rerun()
        video_name = "Webcam"
    else:
        temp_path = Path("temp_video.mp4")
        temp_path.write_bytes(video_file.read())
        cap = cv2.VideoCapture(str(temp_path))
        video_name = video_file.name

    st.markdown(f"**Video source:** `{video_name}`")

    # --- Initialization for counting logic ---
    counter: defaultdict[str, int] = defaultdict(int)
    track_last_pos = {}  # Store last known (cx, cy) for each track ID.

    frame_slot = st.empty()

    try:
        with st.spinner("Processing… (press ⏹️ to stop)"):
            while cap.isOpened():
                # Check for the stop signal at the beginning of each loop.
                if st.session_state.stop_flag:
                    st.warning("Processing stopped by user.")
                    break

                ret, frame = cap.read()
                if not ret:
                    st.success("Video processing complete.")
                    break

                # --- Detection and Tracking ---
                results = model(frame, size=640)
                detections = results.xyxy[0].cpu().numpy()
                dsort_inp = [
                    (
                        [float(b[0]), float(b[1]), float(b[2]), float(b[3])],  # Bounding box
                        float(b[4]),  # Confidence
                        int(b[5]),  # Class ID
                    )
                    for b in detections
                ]
                tracks = tracker.update_tracks(dsort_inp, frame=frame)

                # --- Counting and Drawing ---
                line_y = frame.shape[0] // 2
                line_x = frame.shape[1] // 2

                for trk in tracks:
                    # Skip unconfirmed tracks or those not updated in this frame.
                    if not trk.is_confirmed() or trk.time_since_update > 0:
                        continue

                    tid = trk.track_id
                    x1, y1, x2, y2 = map(int, trk.to_ltrb())
                    cls_id = trk.get_det_class()
                    cls_nm = model.names[cls_id]
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    # Draw bounding box and track info
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{cls_nm} #{tid}",
                        (x1, y1 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)

                    # --- Crossing Logic ---
                    if tid in track_last_pos:
                        last_cx, last_cy = track_last_pos[tid]
                        if flow_direction == "Left to Right" and last_cx < line_x <= cx:
                            counter[cls_nm] += 1
                        elif flow_direction == "Right to Left" and last_cx > line_x >= cx:
                            counter[cls_nm] += 1
                        elif flow_direction == "Up to Down" and last_cy < line_y <= cy:
                            counter[cls_nm] += 1
                        elif flow_direction == "Down to Up" and last_cy > line_y >= cy:
                            counter[cls_nm] += 1

                    track_last_pos[tid] = (cx, cy)

                # Draw counting line
                if flow_direction in ("Left to Right", "Right to Left"):
                    cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (255, 0, 0), 2)
                else:
                    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (255, 0, 0), 2)

                # Update UI every frame
                frame_slot.image(frame[:, :, ::-1], channels="RGB")
                render_counts(counter_slot, counter, f"📊 Live Counts ({flow_direction})")

    finally:
        # --- Cleanup ---
        cap.release()
        if temp_path and temp_path.exists():
            temp_path.unlink()  # Delete temporary video file

        # Reset flags to indicate processing has finished
        st.session_state.processing = False
        st.session_state.stop_flag = False

    # --- Final summary ---
    # Show final counts in both the live counter box and a separate summary area
    render_counts(counter_slot, counter, f"📊 Final Counts ({flow_direction})")
    render_counts(st, counter, "📜 Summary of All Crossings")

    st.success("Finished!")

    # **THE FIX**: Rerun the script to update the UI (e.g., re-enable Start button).
    # This is crucial for resetting the button states after the processing loop has finished.
    st.rerun()

