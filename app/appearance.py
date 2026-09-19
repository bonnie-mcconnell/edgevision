"""
Pixel-based appearance descriptors for identity matching (re-identifying
a track across an occlusion gap). Pixels into numerical score.
"""

from __future__ import annotations

import cv2
import numpy as np


CROP_MARGIN = 0.2 


def appearance_descriptor(frame: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray | None:
    """
    Lightweight visual descriptor for a track: normalized hue histogram
    of the pixels inside its box. Computed in HSV not BGR as it's more 
    stable under brightness changes.

    Simple for now with no body segmentation, no learned embedding.
    Returns None if box is degenerate (fully outside of frame, or zero-area
    after clipping), because there's no pixel content to describe.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box

    # clip to valid frame bounds, convert to int
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))

    if x2 <= x1 or y2 <= y1:
        return None

    # shrink now valid box inward to 60% of its original box
    margin_x = int((x2-x1) * CROP_MARGIN)
    margin_y = int((y2-y1) * CROP_MARGIN)

    x1, x2 = x1 + margin_x, x2 - margin_x
    y1, y2 = y1 + margin_y, y2 - margin_y

    if x2 <= x1 or y2 <= y1:
        return None

    region = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [32], [0, 180])

    # normalize so histogram sums to 1 (a probability distribution)
    cv2.normalize(hist, hist, 1, 0, cv2.NORM_L1)
    return hist


def appearance_distance(hist_a: np.ndarray | None, hist_b: np.ndarray | None) -> float:
    """
    Bhattacharyya distance between two appearance histograms: 0.0 = identical
    distributions, 1.0 = no overlap. Returns the maximum distance (1.0) if 
    either descriptor is missing, because a missing descriptor should never
    win against a real one.
    """
    if hist_a is None or hist_b is None:
        return 1.0
    return cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_BHATTACHARYYA)