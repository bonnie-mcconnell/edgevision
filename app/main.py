"""FastAPI app: runs detection on a video source, alerts on zone hits, streams over ws."""

from __future__ import annotations

import asyncio
import csv
import os
import base64

import cv2
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, UploadFile, File
from fastapi.responses import FileResponse, Response
from contextlib import asynccontextmanager
import numpy as np

from app.alerts import AlertManager, Zone
from app.dependencies import get_alert_manager, get_detector, get_redis_client, get_frame_source
from app.detector import HogPersonDetector, OnnxPersonDetector, draw_detections


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_detector()
    redis_client = get_redis_client()
    redis_client.ping() # raises redis.ConnectionError immediately if Redis is unreachable
    yield


app = FastAPI(title="EdgeVision", lifespan=lifespan)

# Example zone: middle third of a 640x480 frame.
# TODO: In a real deployment this comes from a per-camera config, drawn by the user in a setup UI.
DEFAULT_ZONE = Zone(name="front_door", x1=200, y1=0, x2=440, y2=480)


@app.get("/health")
def health(detector: HogPersonDetector | OnnxPersonDetector = Depends(get_detector)) -> dict:
    return {"status": "ok", "detector_backend": type(detector).__name__}


@app.get("/live")
def live_view() -> FileResponse:
    return FileResponse("app/static/live.html")


@app.get("/alerts/recent")
def recent_alerts(limit: int = 20, alert_manager: AlertManager = Depends(get_alert_manager)) -> dict:
    return {"alerts": alert_manager.recent_alerts(limit)}


@app.get("/benchmark/results")
def benchmark_results() -> dict:
    """Serves the CSV produced by scripts/benchmark.py."""
    path = "results/benchmark.csv"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No benchmark results yet. Run scripts/benchmark.py first.")
    with open(path) as f:
        return {"results": list(csv.DictReader(f))}


@app.post("/demo/detect")
async def demo_detect(
    file: UploadFile = File(...),
    format: str = "image", # or 'json'
    detector: HogPersonDetector | OnnxPersonDetector = Depends(get_detector),
):
    """Post an image to this endpoint, get back detections as JSON or an annotated image."""
    im_bytes = await file.read()
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)
    frame = cv2.imdecode(im_arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    detections, elapsed = detector.detect(frame) 

    if format == "json":
        return {
            "detections": [
                {"label": d.label, "confidence": d.confidence,
                 "box": [d.x1, d.y1, d.x2, d.y2]}
                for d in detections
            ],
            "inference_ms": round(elapsed * 1000, 2),
        }

    frame = draw_detections(frame, detections)

    ok_enc, buffer = cv2.imencode(".jpg", frame)
    if not ok_enc:
        raise HTTPException(status_code=500, detail="Could not encode result image")
    
    return Response(content=buffer.tobytes(), media_type="image/jpeg")




@app.websocket("/ws/detections")
async def websocket_detections(
    websocket: WebSocket, 
    include_frame: bool = False,
    alert_manager: AlertManager = Depends(get_alert_manager), 
    detector: HogPersonDetector | OnnxPersonDetector = Depends(get_detector), 
    cap: cv2.VideoCapture = Depends(get_frame_source)
    ):
    """Streams detections+alerts. Needs VIDEO_SOURCE set to a webcam index or RTSP url."""
    await websocket.accept()

    try:
        if not cap.isOpened():
            await websocket.send_json({"error": f"Could not open video source"})
            return

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            try:
                detections, elapsed_s = detector.detect(frame)
            except Exception as e:
                print(f"detector error on frame, skipping: {e}")
                continue

            fired_alerts = []
            for det in detections:
                if DEFAULT_ZONE.overlaps_box(det.x1, det.y1, det.x2, det.y2):
                    alert = alert_manager.raise_if_new(DEFAULT_ZONE.name, det.label, det.confidence)
                    if alert:
                        fired_alerts.append(alert.to_dict())

            payload = {
                "detections": [
                    {"label": d.label, "confidence": d.confidence,
                     "box": [d.x1, d.y1, d.x2, d.y2]}
                    for d in detections
                ],
                "inference_ms": round(elapsed_s * 1000, 2),
                "alerts": fired_alerts,
            }

            if include_frame:
                ok_enc, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                if ok_enc:
                    payload["frame"] = base64.b64encode(buffer.tobytes()).decode("utf-8")

            await websocket.send_json(payload)
            await asyncio.sleep(0.01)  # yield control, don't peg the event loop
    except WebSocketDisconnect:
        pass
    finally:
        cap.release()
