"""FastAPI app: runs detection on a video source, alerts on zone hits, streams over ws."""

from __future__ import annotations

import asyncio
import csv
import os
import base64
import time

import cv2
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, UploadFile, File
from fastapi.responses import FileResponse, Response
from contextlib import asynccontextmanager
import numpy as np

from app.alerts import AlertManager, default_zone_for_resolution
from app.dependencies import get_alert_manager, get_detector, get_redis_client, get_frame_source
from app.detector import HogPersonDetector, OnnxPersonDetector
from app.drawing import draw_detections
from app.tracker import Tracker, centroid_max_dist_for_resolution


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_detector()
    redis_client = get_redis_client()
    redis_client.ping() # raises redis.ConnectionError immediately if Redis is unreachable
    yield


app = FastAPI(title="EdgeVision", lifespan=lifespan)


# TODO: In a real deployment this comes from a per-camera config, drawn by the user in a setup UI.
# actual zone is computed per connection in websocket_detections, scaled to that connection's resolution


@app.get("/health")
def health(detector: HogPersonDetector | OnnxPersonDetector = Depends(get_detector)) -> dict:
    return {"status": "ok", "detector_backend": type(detector).__name__}


@app.get("/live")
def live_view() -> FileResponse:
    return FileResponse("app/static/live.html")


@app.get("/demo")
def demo_view() -> FileResponse:
    return FileResponse("app/static/demo.html")


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
    t0 = time.perf_counter()
    im_bytes = await file.read()
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)
    frame = cv2.imdecode(im_arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    t1 = time.perf_counter()

    detections, detect_elapsed = detector.detect(frame) 
    t2 = time.perf_counter()

    timing = {
        "decode_ms": round((t1-t0) * 1000, 2),
        "detect_ms": round(detect_elapsed * 1000, 2)
    }

    if format == "json":
        return {
            "detections": [
                {"label": d.label, "confidence": d.confidence,
                 "box": [d.x1, d.y1, d.x2, d.y2]}
                for d in detections
            ],
            "timing": timing,
        }

    frame = draw_detections(frame, detections)

    ok_enc, buffer = cv2.imencode(".jpg", frame)
    if not ok_enc:
        raise HTTPException(status_code=500, detail="Could not encode result image")

    t3 = time.perf_counter()
    timing["draw_and_encode_ms"] = round((t3-t2) * 1000, 2)

    print(f"/demo/detect timing: {timing}") # visible server side even for image response, which doesn't have JSON body
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

        frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        tracker = Tracker(centroid_max_dist=centroid_max_dist_for_resolution(frame_width, frame_height))
        zone = default_zone_for_resolution(frame_width, frame_height)

        while True:
            read_start = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            read_done = time.perf_counter()

            try:
                detections, elapsed_s = detector.detect(frame)
            except Exception as e:
                print(f"detector error on frame, skipping: {e}")
                continue
            detect_done = time.perf_counter()

            tracks = tracker.update(detections)
            track_done = time.perf_counter()

            fired_alerts = []
            for track in tracks:
                if zone.overlaps_box(*track.box):
                    alert = alert_manager.raise_if_new(zone.name, track.label, track.confidence)
                    if alert:
                        fired_alerts.append(alert.to_dict())

            payload = {
                "detections": [
                    {"track_id": t.track_id, "label": t.label, "confidence": t.confidence, "box": list(t.box)}
                    for t in tracks
                ],
                "zone": {"name": zone.name, "x1": zone.x1, "y1": zone.y1, "x2": zone.x2, "y2": zone.y2},
                "inference_ms": round(elapsed_s * 1000, 2),
                "alerts": fired_alerts,
                "timing": {       # inference_ms kept for backward compat with existing consumers, timing gives the fuller breakdown
                    "read_ms": round((read_done - read_start) * 1000, 2),
                    "detect_ms": round(elapsed_s * 1000, 2),
                    "track_ms": round((track_done - detect_done) * 100, 2),
                },
            }

            if include_frame:
                encode_start = time.perf_counter()
                ok_enc, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                if ok_enc:
                    payload["frame"] = base64.b64encode(buffer.tobytes()).decode("utf-8")
                payload["timing"]["encode_ms"] = round((time.perf_counter() - encode_start) * 1000, 2)

            await websocket.send_json(payload)
            await asyncio.sleep(0.01)  # yield control, don't peg the event loop
    except WebSocketDisconnect:
        pass
    finally:
        cap.release()