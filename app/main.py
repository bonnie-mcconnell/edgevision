"""FastAPI app: runs detection on a video source, alerts on zone hits, streams over ws."""

from __future__ import annotations

import asyncio
import csv
import os

import cv2
import redis
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from app.alerts import AlertManager, Zone
from app.detector import build_detector

app = FastAPI(title="EdgeVision")

redis_client = redis.Redis(
    host=os.environ.get("REDIS_HOST", "localhost"),
    port=int(os.environ.get("REDIS_PORT", 6379)),
    decode_responses=True,
)
alert_manager = AlertManager(redis_client, cooldown_seconds=30)
detector = build_detector()

# Example zone: middle third of a 640x480 frame.
# TODO: In a real deployment this comes from a per-camera config, drawn by the user in a setup UI.
DEFAULT_ZONE = Zone(name="front_door", x1=200, y1=0, x2=440, y2=480)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "detector_backend": type(detector).__name__}


@app.get("/alerts/recent")
def recent_alerts(limit: int = 20) -> dict:
    return {"alerts": alert_manager.recent_alerts(limit)}


@app.get("/benchmark/results")
def benchmark_results() -> dict:
    """Serves the CSV produced by scripts/benchmark.py."""
    path = "results/benchmark.csv"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No benchmark results yet. Run scripts/benchmark.py first.")
    with open(path) as f:
        return {"results": list(csv.DictReader(f))}


@app.websocket("/ws/detections")
async def websocket_detections(websocket: WebSocket):
    """Streams detections+alerts. Needs VIDEO_SOURCE set to a webcam index or RTSP url."""
    await websocket.accept()

    source = os.environ.get("VIDEO_SOURCE", "0")
    source = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(source)

    try:
        if not cap.isOpened():
            await websocket.send_json({"error": f"Could not open video source: {source}"})
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

            await websocket.send_json({
                "detections": [
                    {"label": d.label, "confidence": d.confidence,
                     "box": [d.x1, d.y1, d.x2, d.y2]}
                    for d in detections
                ],
                "inference_ms": round(elapsed_s * 1000, 2),
                "alerts": fired_alerts,
            })
            await asyncio.sleep(0.01)  # yield control, don't peg the event loop
    except WebSocketDisconnect:
        pass
    finally:
        cap.release()
