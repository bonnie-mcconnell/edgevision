"""
Diagnostic script for the track swapping: when a track dies
does a new track get born right afterwards in roughly the same place, 
suggesting that it's the same physical person getting a new ID, not a
person who's actually left? 
Includes before/after validation of Hungarian vs Greedy tracker by
running pipeline with each matching algorithm & everything else fixed
and comparing fragmentation rates.
"""
import math
import os
import csv

import cv2

from app.detector import OnnxDetector
from app.tracker import Tracker, _greedy_match, _hungarian_match, TUNED_IOU_THRESHOLD, TUNED_MAX_AGE


VIDEO_SOURCE = "test_footage/street.mp4"

# how close in time/space counts as "plausibly the same person, re-spawned"
# widened from original 15 frame check
REASSOC_MAX_FRAMES = 45   # ~1.5s at 30fps
REASSOC_MAX_DIST = 150.0  # pixels, at this footage's 1920x1080 resolution


def centroid(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def match_deaths_to_births(death_events, birth_events, max_frames=REASSOC_MAX_FRAMES, max_dist=REASSOC_MAX_DIST) -> list[dict]:
    """For each death, decide whether a nearby birth is is a respawn of the same track.
    Closest pairs first matching."""
    candidates = [] # (distance, death_index, birth_index)
    for di, (_died_id, death_frame, death_box) in enumerate(death_events):
        dc = centroid(death_box)
        for bi, (_born_id, birth_frame, birth_box) in enumerate(birth_events):
            if death_frame < birth_frame <= death_frame + max_frames:
                dist = math.dist(dc, centroid(birth_box))
                if dist <= max_dist:
                    candidates.append((dist, di, bi))
    candidates.sort(key=lambda c: c[0])

    matched_birth_for_death: dict[int, tuple[int, float]] = {}
    claimed_births: set[int] = set()
    for dist, di, bi in candidates:
        if di in matched_birth_for_death or bi in claimed_births:
            continue
        matched_birth_for_death[di] = (bi, dist)
        claimed_births.add(bi)

    rows = []
    for di, (died_id, death_frame, _death_box) in enumerate(death_events):
        match = matched_birth_for_death.get(di)
        if match is None:
            rows.append({
                "died_id": died_id, "death_frame": death_frame, "reassociated": False,
                "born_id": None, "birth_frame": None, "gap_frames": None, "distance_px": None,
            })
            continue
        bi, dist = match
        born_id, birth_frame, _birth_box = birth_events[bi]
        rows.append({
            "died_id": died_id, "death_frame": death_frame, "reassociated": True,
            "born_id": born_id, "birth_frame": birth_frame, "gap_frames": birth_frame - death_frame,
            "distance_px": round(dist, 1),
        })
    return rows


def run_fragmentation_analysis(match_fn, out_csv: str, revival_csv: str) -> dict:
    """Runs the full detect -> track -> fragmentation-check pipeline once,
    using the given matching function. Returns summary stats so a caller
    can run this twice (once per match_fn) and compare directly."""
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise SystemExit(f"Couldn't open {VIDEO_SOURCE}")

    detector = OnnxDetector("models/yolov8n.onnx")
    revival_log: list[dict] = []
    tracker = Tracker(match_fn=match_fn, revival_log=revival_log, iou_threshold=TUNED_IOU_THRESHOLD, max_age=TUNED_MAX_AGE)

    death_events = []  # (track_id, death_frame, last_box)
    birth_events = []  # (track_id, birth_frame, first_box)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        detections, _ = detector.detect(frame)

        before = {t.track_id: t.box for t in tracker.tracks}
        tracker.update(detections, frame=frame)
        after = {t.track_id: t.box for t in tracker.tracks}

        for tid in (set(before) - set(after)):
            death_events.append((tid, frame_idx, before[tid]))
        for tid in (set(after) - set(before)):
            birth_events.append((tid, frame_idx, after[tid]))

        if frame_idx % 60 == 0:
            print(f"  frame {frame_idx}")

    cap.release()

    rows = match_deaths_to_births(death_events, birth_events)
    plausible = sum(1 for r in rows if r["reassociated"])

    os.makedirs("results", exist_ok=True)
    if rows:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    if revival_log:
        with open(revival_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=revival_log[0].keys())
            writer.writeheader()
            writer.writerows(revival_log)
    print(f"  {len(revival_log)} revival candidate comparisons logged, "
          f"{sum(1 for r in revival_log if r['passed'])} passed threshold "
          f"({revival_csv})")

    pct = 100 * plausible / len(death_events) if death_events else 0
    return {
        "deaths": len(death_events),
        "births": len(birth_events),
        "frames": frame_idx,
        "plausible_fragmentations": plausible,
        "fragmentation_pct": pct,
    }


def main() -> None:
    print("Running with greedy matching (the original tracker):")
    greedy_stats = run_fragmentation_analysis(_greedy_match, "results/fragmentation_events_greedy.csv", "results/revival_attempts_greedy.csv")

    print("\nRunning with hungarian matching (the new optimal-assignment tracker):")
    hungarian_stats = run_fragmentation_analysis(_hungarian_match, "results/fragmentation_events_hungarian.csv", "results/revival_attempts_hungarian.csv")

    print(f"\n{'':25s} {'greedy':>12s} {'hungarian':>12s}")
    for key in ("deaths", "births", "plausible_fragmentations"):
        print(f"{key:25s} {greedy_stats[key]:>12} {hungarian_stats[key]:>12}")
    print(f"{'fragmentation %':25s} {greedy_stats['fragmentation_pct']:>11.1f}% {hungarian_stats['fragmentation_pct']:>11.1f}%")
    print(f"\n(within {REASSOC_MAX_FRAMES} frames, {REASSOC_MAX_DIST}px of a death, a nearby birth is likely a fragmented identity, not a real exit)")

    delta = hungarian_stats["plausible_fragmentations"] - greedy_stats["plausible_fragmentations"]
    if delta < 0:
        print(f"Hungarian reduced plausible fragmentations by {-delta} vs greedy on this clip.")
    elif delta > 0:
        print(f"Hungarian did NOT reduce fragmentation here - {delta} MORE plausible fragmentations than greedy on this clip.")
    else:
        print("No difference in plausible fragmentations between greedy and hungarian on this clip.")
    print("Compare the two CSVs by death_frame (not died_id, track IDs are relabeled independently per run) to see which specific events actually differ.")


if __name__ == "__main__":
    main()