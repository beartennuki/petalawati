# vehicle_counter_yolov8.py  (fixed COCO-ID mismatch)
# ─────────────────────────────────────────────────────────────────────────────
# Detects and counts only cars, motorcycles, buses and trucks.
# Uses an explicit ID→name map to avoid class-index mix-ups.
# ─────────────────────────────────────────────────────────────────────────────
# Requirements: streamlit, ultralytics, opencv-python, deep-sort-realtime
# Tested: Python 3.10 / Streamlit 1.34 / Ultralytics 8.x
# ----------------------------------------------------------------------------
import streamlit as st
import cv2
from pathlib import Path
from collections import defaultdict
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

# ────────────────── UI ──────────────────
st.set_page_config(page_title="Vehicle Counter (YOLOv8)")
st.title("🚗 Vehicle Counter (YOLOv8)")

if "stop_flag" not in st.session_state:
    st.session_state.stop_flag = False
if "processing" not in st.session_state:
    st.session_state.processing = False

video_file = st.file_uploader(
    "Upload a video (MP4/AVI) or leave blank to use webcam",
    type=["mp4", "avi"],
)
conf_thres = st.slider("YOLO confidence threshold", 0.1, 1.0, 0.4, 0.05)

flow_direction = st.selectbox(
    "Select Flow Direction",
    ("Left to Right", "Right to Left", "Up to Down", "Down to Up"),
)

counter_slot = st.empty()
cols = st.columns([1, 1])
start_button = cols[0].button("▶️ Start", type="primary", disabled=st.session_state.processing)
stop_button  = cols[1].button("⏹️ Stop", disabled=not st.session_state.processing)
if stop_button:
    st.session_state.stop_flag = True

# ────────────────── Model + Tracker ──────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    model = YOLO("yolov8s.pt")        # small model; swap to *n.pt for edge devices
    tracker = DeepSort(
        max_age=10, n_init=3, max_iou_distance=0.7,
        nms_max_overlap=1.0, embedder="mobilenet", half=True,
    )
    return model, tracker

# ────────────────── Vehicle classes (COCO 0-based) ──────────────────
# 0:person 1:bicycle **2:car 3:motorcycle 4:airplane 5:bus 6:train 7:truck**
LABEL_MAP = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
VEHICLE_CLASSES = list(LABEL_MAP.keys())   # [2, 3, 5, 7]

def render_counts(container, counts, title):
    with container.container():
        st.subheader(title)
        if counts:
            for cls, n in sorted(counts.items()):
                st.write(f"**{cls}** : {n}")
        else:
            st.write("_No crossings detected._")

# ────────────────── Main loop ──────────────────
if start_button:
    st.session_state.stop_flag = False
    st.session_state.processing = True
    st.rerun()

if st.session_state.get("processing", False):
    model, tracker = load_models()

    # video capture
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

    counter             = defaultdict(int)
    track_last_pos      = {}
    MAX_BOX_H_RATIO     = 0.40   # ignore boxes taller than 40 % of frame height
    frame_slot          = st.empty()

    try:
        with st.spinner("Processing… (press ⏹️ to stop)"):
            while cap.isOpened():
                if st.session_state.stop_flag:
                    st.warning("Processing stopped by user.")
                    break

                ok, frame = cap.read()
                if not ok:
                    st.success("Video processing complete.")
                    break

                # ── Detection (filter by class + conf) ──
                results = model.predict(
                    frame, classes=VEHICLE_CLASSES, conf=conf_thres, verbose=False
                )
                detections = []
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    w, h           = x2 - x1, y2 - y1
                    conf           = box.conf.item()
                    cls_id         = int(box.cls.item())
                    detections.append(([x1, y1, w, h], conf, cls_id))

                # ── DeepSort tracking ──
                tracks = tracker.update_tracks(detections, frame=frame)

                H, W = frame.shape[:2]
                line_y = H // 2
                line_x = W // 2

                for trk in tracks:
                    if not trk.is_confirmed() or trk.time_since_update > 0:
                        continue

                    x1, y1, x2, y2 = map(int, trk.to_ltrb())
                    if (y2 - y1) > H * MAX_BOX_H_RATIO:
                        continue

                    tid        = trk.track_id
                    cls_id     = trk.get_det_class()
                    cls_name   = LABEL_MAP.get(cls_id, f"id{cls_id}")
                    cx, cy     = (x1 + x2) // 2, (y1 + y2) // 2

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{cls_name} #{tid}", (x1, y1 - 7),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)

                    # crossing test
                    if tid in track_last_pos:
                        pcx, pcy = track_last_pos[tid]
                        if   flow_direction == "Left to Right" and pcx < line_x <= cx:
                            counter[cls_name] += 1
                        elif flow_direction == "Right to Left" and pcx > line_x >= cx:
                            counter[cls_name] += 1
                        elif flow_direction == "Up to Down"   and pcy < line_y <= cy:
                            counter[cls_name] += 1
                        elif flow_direction == "Down to Up"   and pcy > line_y >= cy:
                            counter[cls_name] += 1
                    track_last_pos[tid] = (cx, cy)

                # draw counting line
                if flow_direction in ("Left to Right", "Right to Left"):
                    cv2.line(frame, (line_x, 0), (line_x, H), (255, 0, 0), 2)
                else:
                    cv2.line(frame, (0, line_y), (W, line_y), (255, 0, 0), 2)

                frame_slot.image(frame[:, :, ::-1], channels="RGB")
                render_counts(counter_slot, counter, f"📊 Live Counts ({flow_direction})")

    finally:
        cap.release()
        if temp_path and temp_path.exists():
            temp_path.unlink()
        st.session_state.processing = False
        st.session_state.stop_flag  = False

    render_counts(counter_slot, counter, f"📊 Final Counts ({flow_direction})")
    render_counts(st,           counter, "📜 Summary of All Crossings")
    st.success("Finished!")
    st.rerun()
