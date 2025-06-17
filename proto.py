# vehicle_counter_app.py
#
# Streamlit demo – Vehicle counting with a *vertical* centre-line
# Requirements: streamlit, torch, torchvision, opencv-python, deep-sort-realtime
# Tested with Python 3.10 / Streamlit 1.34 / Torch 2.3

import streamlit as st
import torch
import cv2
from pathlib import Path
from collections import defaultdict
from deep_sort_realtime.deepsort_tracker import DeepSort

# ──────────────────────────────  Streamlit UI  ──────────────────────────────
st.set_page_config(page_title="Vehicle Counter – vertical line", layout="wide")
st.title("🚗 Vehicle Counter (vertical line at frame centre)")

video_file = st.file_uploader(
    "Upload a video (MP4/AVI) or leave blank to use webcam",
    type=["mp4", "avi"],
)
conf_thres = st.slider("YOLO confidence threshold", 0.1, 1.0, 0.35, 0.05)
run_btn = st.button("▶️ Start")

# ─────────────────────────────  Load models  ────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models(user_conf):
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    model.conf = user_conf
    model.classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

    tracker = DeepSort(
        max_age=30,
        n_init=2,
        max_iou_distance=0.7,
        nms_max_overlap=1.0,
        embedder="mobilenet",
        half=True,
    )
    return model, tracker

# ───────────────────────────────  Main loop  ────────────────────────────────
if run_btn:
    model, tracker = load_models(conf_thres)

    # open capture
    if video_file is None:
        cap = cv2.VideoCapture(0)
        video_name = "Webcam"
    else:
        temp_path = Path("temp_video.mp4")
        temp_path.write_bytes(video_file.read())
        cap = cv2.VideoCapture(str(temp_path))
        video_name = video_file.name

    st.markdown(f"**Video source:** `{video_name}`")

    # counters & caches
    counter          = defaultdict(int)   # {class_name: total}
    track_last_x     = {}                 # {track_id: last centre-x}

    frame_slot = st.empty()

    with st.spinner("Processing…"):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # ── Inference ────────────────────────────────────────────────────
            results = model(frame, size=640)
            detections = results.xyxy[0].cpu().numpy()  # n × 6  (x1,y1,x2,y2,conf,cls)

            # format detections for DeepSORT
            dsort_inp = [
                [
                    [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                    float(b[4]),
                    int(b[5]),
                ]
                for b in detections
            ]

            tracks = tracker.update_tracks(dsort_inp, frame=frame)

            # centre of vertical counting line
            line_x = frame.shape[1] // 2

            # ── Draw & count ────────────────────────────────────────────────
            for trk in tracks:
                if not trk.is_confirmed():
                    continue
                tid     = trk.track_id
                x1, y1, x2, y2 = map(int, trk.to_ltrb())
                cls_id  = int(trk.det_class)
                cls_nm  = model.names[cls_id]

                # object centre
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # draw bbox & id
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{cls_nm} #{tid}", (x1, y1 - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)

                # count when crossing from left → right
                if tid in track_last_x and track_last_x[tid] < line_x <= cx:
                    counter[cls_nm] += 1
                track_last_x[tid] = cx

            # vertical counting line
            cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (255, 0, 0), 2)

            # show frame
            frame_slot.image(frame[:, :, ::-1], channels="RGB")

    cap.release()

    # ──────────────────────────────  Results  ───────────────────────────────
    st.subheader("📊 Vehicle counts (left → right crossings)")
    if counter:
        for cls, n in counter.items():
            st.write(f"**{cls}** : {n}")
    else:
        st.write("_No crossings detected._")
    st.success("Finished!")

