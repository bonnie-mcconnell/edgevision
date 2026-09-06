"""
Diagnostic script for the track swapping: when a track dies
does a new track get born right afterwards in roughly the same place, 
suggesting that it's the same physical person getting a new ID, not a
person who's actually left?
"""
import math
import os
import csv

import cv2

from app.detector import OnnxPersonDetector
from app.tracker import Tracker


VIDEO_SOURCE = "test_footage/street.mp4"
OUT_CSV = "results/fragmentation_events.csv"

# how close in time/space counts as "plausibly the same person, re-spawned"
# widened from original 15 frame check
REASSOC_MAX_FRAMES = 45   # ~1.5s at 30fps
REASSOC_MAX_DIST = 150.0  # pixels, at this footage's 1920x1080 resolution


def centroid(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise SystemExit(f"Couldn't open {VIDEO_SOURCE}")

    detector = OnnxPersonDetector("models/yolov8n.onnx")
    tracker = Tracker()

    death_events = [] # (track_id, death_frame, last_box)
    birth_events = [] # (track_id, birth_frame, first_box)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        detections, _ = detector.detect(frame)

        before = {t.track_id: t.box for t in tracker.tracks}
        tracker.update(detections)
        after = {t.track_id: t.box for t in tracker.tracks}

        for tid in (set(before) - set(after)):
            death_events.append((tid, frame_idx, before[tid]))
        for tid in (set(after) - set(before)):
            birth_events.append((tid, frame_idx, after[tid]))

        if frame_idx % 60 == 0:
            print(f"frame {frame_idx}")

    cap.release()

    (f"\n{len(death_events)} deaths, {len(birth_events)} births over {frame_idx} frames")

    plausible = 0
    rows = []
    for died_id, death_frame, death_box in death_events:
        dc = centroid(death_box)
        best_born_id, best_birth_frame, best_dist = None, None, None
        for born_id, birth_frame, birth_box in birth_events:
            if death_frame < birth_frame <= death_frame + REASSOC_MAX_FRAMES:
                dist = math.dist(dc, centroid(birth_box))
                if dist <= REASSOC_MAX_DIST and (best_dist is None or dist < best_dist):
                    best_born_id, best_birth_frame, best_dist = born_id, birth_frame, dist

        flagged = best_born_id is not None
        if flagged:
            plausible += 1
        rows.append({
            "died_id": died_id,
            "death_frame": death_frame,
            "reassociated": flagged,
            "born_id": best_born_id,
            "birth_frame": best_birth_frame,
            "gap_frames": (best_birth_frame - death_frame) if flagged else None,
            "distance_px": round(best_dist, 1) if flagged else None, # type: ignore
        })

    os.makedirs("results", exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    pct = 100 * plausible / len(death_events) if death_events else 0
    print(f"\ndeaths with a plausible same-person rebirth nearby: {plausible}/{len(death_events)} ({pct:.1f}%)")
    print(f"(within {REASSOC_MAX_FRAMES} frames, {REASSOC_MAX_DIST}px, they're likely fragmented identities not real exits)")
    print(f"wrote per-death detail to {OUT_CSV}")

if __name__ == "__main__":
    main()