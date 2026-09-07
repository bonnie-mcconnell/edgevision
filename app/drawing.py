"""
All frame-annotation/visualization code, separate
from app/detector.py (inference logic) and app/tracker.py (matching logic,
kept dependency-light). Anything that draws on frames goes here.
"""

from __future__ import annotations

import colorsys

import cv2
import numpy as np

from app.detector import Detection
from app.tracker import Track


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """
    Draws a box + label/confidence text for each detection directly onto
    frame (mutates it in place, same object is also returned for convenience
    when chaining). Shared by scripts/test_video.py, the /demo/detect route,
    and anywhere else that needs the same visual style, so it only needs to
    be changed in one place.
    """
    for d in detections:
        x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, f"{d.label} {d.confidence:.2f}", (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame


def _color_for_track(track_id: int) -> tuple[int, int, int]:
    """
    Deterministic BGR color per track ID for cv2 drawing, via golden-angle hue
    rotation (same as live.html's colorForTrack, ported for OpenCV: HSL
    in a browser vs. Python's colorsys HLS, plus an RGB->BGR channel swap).
    Consecutive integer IDs land far apart on the color wheel with no setup/hardcoded palette.
    """
    hue = ((track_id * 137.5) % 360) / 360.0
    r, g, b = colorsys.hls_to_rgb(hue, 0.55, 0.85)  # colorsys param order is (h, l, s)
    return (int(b * 255), int(g * 255), int(r * 255))


def draw_tracks(
        frame: np.ndarray, 
        tracks: list[Track], 
        dwell_seconds: dict[int, float] | None = None,
        stationary_seconds: dict[int, float] | None = None,
        loitering_threshold: float | None = None,
    ) -> np.ndarray:
    """
    Draws each confirmed track's box + track_id/label/confidence, color-coded
    by track_id so the same person keeps the same color across frames,
    including across a gap where their box briefly disappeared. Mutates
    frame in place (same object also returned).

    If given, dwell_seconds/stationary_seconds are {track_id: seconds} maps computed by caller, 
    appended to the label when present. stationary_seconds should only include tracks
    where tracker.has_moved(t) is True. If loitering_threshold is given and a
    track's stationary time meets or exceeds it, the label is drawn in red
    with a "LOITERING" flag instead of its usual track color.
    """
    for t in tracks:
        x1, y1, x2, y2 = int(t.box[0]), int(t.box[1]), int(t.box[2]), int(t.box[3])
        color = _color_for_track(t.track_id)
        label = f"#{t.track_id} {t.label} {t.confidence:.2f}"

        if dwell_seconds is not None and t.track_id in dwell_seconds:
            label += f" | {dwell_seconds[t.track_id]:.1f}s"

        is_loitering = False
        if stationary_seconds is not None and t.track_id in stationary_seconds:
            stat = stationary_seconds[t.track_id]
            label += f" | still {stat:.1f}s"
            if loitering_threshold is not None and stat >= loitering_threshold:
                is_loitering = True
                label += " LOITERING"

        box_color = (0, 0, 255) if is_loitering else color
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
        cv2.putText(frame, label, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return frame