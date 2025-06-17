# main.py
import cv2
import uvicorn
import asyncio
import uuid
import json
from fastapi import FastAPI, Request, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from collections import defaultdict
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
from pathlib import Path

# --- Basic Setup ---
app = FastAPI(title="FastAPI Vehicle Counter")
templates = Jinja2Templates(directory="templates")


# --- WebSocket Connection Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()

# --- Model Loading and Caching ---
# A dictionary to cache loaded YOLO models
yolo_models = {}


def get_yolo_model(version="yolov8n.pt"):
    """Loads a YOLO model from cache or from disk."""
    if version not in yolo_models:
        print(f"Loading YOLO model: {version}...")
        yolo_models[version] = YOLO(version)
        print(f"Model {version} loaded and cached.")
    return yolo_models[version]


# --- Global State Management ---
class AppState:
    def __init__(self):
        self.processing = False
        self.stop_signal = False
        self.video_path = None
        self.flow_direction = "Left to Right"
        self.conf_thres = 0.4
        self.counts = defaultdict(int)
        self.yolo_model = None  # Will hold the currently selected model


state = AppState()

# --- DeepSort Tracker ---
tracker = DeepSort(
    max_age=10, n_init=3, max_iou_distance=0.7,
    nms_max_overlap=1.0, embedder="mobilenet", half=True,
)

# --- Vehicle Classes (COCO 0-based) ---
LABEL_MAP = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
VEHICLE_CLASSES = list(LABEL_MAP.keys())


# --- Video Processing Generator ---
async def process_video():
    """
    Processes video, yields frames, and broadcasts count updates via WebSocket.
    """
    if not state.yolo_model:
        print("Error: YOLO model not set in state.")
        return

    cap = cv2.VideoCapture(state.video_path or 0)
    if not cap.isOpened():
        print(f"Error: Could not open video source: {state.video_path or 'webcam'}")
        state.processing = False
        return

    track_last_pos = {}
    MAX_BOX_H_RATIO = 0.40
    state.counts.clear()

    initial_data = json.dumps({"counts": {}, "processing": True})
    await manager.broadcast(initial_data)

    try:
        while not state.stop_signal and cap.isOpened():
            ok, frame = cap.read()
            if not ok: break

            results = state.yolo_model.predict(frame, classes=VEHICLE_CLASSES, conf=state.conf_thres, verbose=False)
            detections = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w, h = x2 - x1, y2 - y1
                conf = box.conf.item()
                cls_id = int(box.cls.item())
                detections.append(([x1, y1, w, h], conf, cls_id))

            tracks = tracker.update_tracks(detections, frame=frame)

            H, W = frame.shape[:2]
            line_y, line_x = H // 2, W // 2

            count_updated = False
            for trk in tracks:
                if not trk.is_confirmed() or trk.time_since_update > 0: continue

                x1, y1, x2, y2 = map(int, trk.to_ltrb())
                if (y2 - y1) > H * MAX_BOX_H_RATIO: continue

                tid, cls_id = trk.track_id, trk.get_det_class()
                cls_name = LABEL_MAP.get(cls_id, f"id{cls_id}")
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{cls_name} #{tid}", (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)

                if tid in track_last_pos:
                    pcx, pcy = track_last_pos[tid]
                    crossed = False
                    if state.flow_direction == "Left to Right" and pcx < line_x <= cx:
                        crossed = True
                    elif state.flow_direction == "Right to Left" and pcx > line_x >= cx:
                        crossed = True
                    elif state.flow_direction == "Up to Down" and pcy < line_y <= cy:
                        crossed = True
                    elif state.flow_direction == "Down to Up" and pcy > line_y >= cy:
                        crossed = True

                    if crossed:
                        state.counts[cls_name] += 1
                        count_updated = True

                track_last_pos[tid] = (cx, cy)

            if count_updated:
                counts_data = json.dumps({"counts": dict(state.counts), "processing": True})
                await manager.broadcast(counts_data)

            if state.flow_direction in ("Left to Right", "Right to Left"):
                cv2.line(frame, (line_x, 0), (line_x, H), (255, 0, 0), 2)
            else:
                cv2.line(frame, (0, line_y), (W, line_y), (255, 0, 0), 2)

            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            await asyncio.sleep(0.01)

    finally:
        cap.release()
        if state.video_path and Path(state.video_path).exists():
            try:
                Path(state.video_path).unlink()
                print(f"Deleted temp video: {state.video_path}")
            except OSError as e:
                print(f"Error deleting temp video: {e}")

        final_data = json.dumps({"counts": dict(state.counts), "processing": False})
        await manager.broadcast(final_data)

        state.processing = False
        state.stop_signal = False
        state.video_path = None
        state.yolo_model = None
        print("Video processing finished and state cleaned up.")


# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.post("/start")
async def start_processing(
        yolo_version: str = Form(...),
        flow_direction: str = Form(...),
        conf_thres: float = Form(...),
        file: UploadFile = File(None)
):
    if state.processing:
        return {"status": "error", "message": "Already processing."}

    state.yolo_model = get_yolo_model(yolo_version)

    if file and file.filename:
        temp_dir = Path("temp_videos")
        temp_dir.mkdir(exist_ok=True)
        unique_id = uuid.uuid4().hex[:8]
        safe_filename = Path(file.filename).name
        video_path = temp_dir / f"{unique_id}_{safe_filename}"
        with open(video_path, "wb") as f:
            f.write(await file.read())
        state.video_path = str(video_path)
    else:
        state.video_path = None

    state.processing = True
    state.stop_signal = False
    state.flow_direction = flow_direction
    state.conf_thres = conf_thres
    print(
        f"Starting processing with: model={yolo_version}, direction={flow_direction}, confidence={conf_thres}, video={state.video_path or 'webcam'}")
    return {"status": "started"}


@app.post("/stop")
async def stop_processing():
    if state.processing:
        state.stop_signal = True
        print("Stop signal sent to processor.")
    return {"status": "stopping"}


@app.get("/video_feed")
async def video_feed():
    if not state.processing:
        return HTMLResponse("Not processing or invalid state.", status_code=400)
    return StreamingResponse(process_video(), media_type="multipart/x-mixed-replace; boundary=frame")
