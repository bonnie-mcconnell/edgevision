"""
Diagnostic script to log the best available match score for 
every track on every frame, even when it falls below threshold 
and the match is rejected, so we can see the internal working of 
tracker/the distribution of frame-to-frame IoU/centroid-distance on 
real footage, to decide on threshold values for both functions
given real jitter/resolution and behaviour of an actual track.
"""

import csv
import os

import cv2

from app.detector import OnnxPersonDetector
from app.tracker import Tracker, _build_score_matrix, _iou, _centroid_dist


VIDEO_SOURCE = "test_footage/street.mp4"
OUT_CSV = "results/tracking_diagnostics.csv"


def main() -> None:
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        abs_path = os.path.abspath(VIDEO_SOURCE)
        raise SystemExit(
            f"Couldn't open {VIDEO_SOURCE}\n"
            f"  resolved to: {abs_path}\n"
            f"  exists on disk: {os.path.exists(VIDEO_SOURCE)}\n"
            f"  current working directory: {os.getcwd()}\n"
        )
    
    detector = OnnxPersonDetector("models/yolov8n.onnx")
    tracker = Tracker()

    os.makedirs("results", exist_ok=True)
    rows = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        detections, _ = detector.detect(frame)

        # log best available score for every existing track
        # against every current detection
        if tracker.tracks and detections:
            iou_matrix = _build_score_matrix(tracker.tracks, detections, _iou)
            cent_matrix = _build_score_matrix(tracker.tracks, detections, _centroid_dist)
            for i, track in enumerate(tracker.tracks):
                best_iou = max(iou_matrix[i]) if iou_matrix[i] else 0.0
                best_cent = min(cent_matrix[i]) if cent_matrix[i] else float('inf')
                rows.append({
                    "frame": frame_idx,
                    "track_id": track.track_id,
                    "track_misses_so_far": track.misses,
                    "best_iou_available":round(best_iou, 4),
                    "best_centroid_dist_available": round(best_cent, 2),
                })

        tracker.update(detections)

        if frame_idx % 60 == 0:
            print(f"frame {frame_idx}")

    cap.release()

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {OUT_CSV}")

if __name__ == "__main__":
    main()