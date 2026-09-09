import math
from collections.abc import Callable
from dataclasses import dataclass

from app.detector import Detection


def centroid_max_dist_for_resolution(width: float, height: float, pct: float = 0.07) -> float:
    """Return a centroid distance threshold scaled to frame size instead of using
    a constant. 7% derived from scripts/diagnose_fragmentation.py."""
    return pct * math.hypot(width, height)


def stationary_move_threshold_for_resolution(width: float, height: float, pct: float = 0.02) -> float:
    """
    Returns the distance a track's centorid must move from it's anchor position to
    be seen as relocation (not standing still jitter). Scaled to frame size.
    """
    return pct * math.hypot(width, height)


@dataclass
class Track:
    track_id: int
    box: tuple[float, float, float, float]
    label: str
    confidence: float
    hits: int
    misses: int
    confirmed: bool
    first_seen_frame: int
    anchor_box: tuple[float, float, float, float] 
    stationary_since_frame: int


class Tracker:
    def __init__(self, iou_threshold: float = 0.3,
                 centroid_max_dist: float = 75.0,
                 min_hits: int = 3,
                 max_age: int = 8,
                 stationary_threshold: float = 20.0):
        self.tracks: list[Track] = []
        self._next_id = 0
        self.iou_threshold = iou_threshold
        self.centroid_max_dist = centroid_max_dist
        self.min_hits = min_hits
        self.max_age = max_age
        self.stationary_threshold = stationary_threshold
        self._frame_count = 0

    def update(self, detections: list[Detection]) -> list[Track]:
        """Call once per frame. Returns currently confirmed tracks."""
        self._frame_count += 1

        iou_matrix = _build_score_matrix(self.tracks, detections, _iou)
        matched, unmatched_tracks, unmatched_det = _greedy_match(
            iou_matrix, self.iou_threshold, higher_better=True, num_cols=len(detections)
        )

        leftover_tracks = [self.tracks[i] for i in unmatched_tracks]
        leftover_dets = [detections[j] for j in unmatched_det]

        centroid_matrix = _build_score_matrix(leftover_tracks, leftover_dets, _centroid_dist)
        matched_r2, still_unmatched_r2_rows, still_unmatched_r2_cols = _greedy_match(
            centroid_matrix, self.centroid_max_dist, higher_better=False, num_cols=len(leftover_dets)
        )

        matched_r2_translated = [(unmatched_tracks[r], unmatched_det[c]) for r, c in matched_r2]
        final_unmatched_tracks = [unmatched_tracks[i] for i in still_unmatched_r2_rows]
        final_unmatched_dets = [unmatched_det[i] for i in still_unmatched_r2_cols]

        matched.extend(matched_r2_translated)

        for track_idx, det_idx in matched:
            track = self.tracks[track_idx]
            det = detections[det_idx]
            track.box = (det.x1, det.y1, det.x2, det.y2)
            track.label = det.label
            track.confidence = det.confidence
            track.hits += 1
            track.misses = 0
            if track.hits >= self.min_hits:
                track.confirmed = True

            if _centroid_dist(track.box, track.anchor_box) >= self.stationary_threshold:
                track.anchor_box = track.box
                track.stationary_since_frame = self._frame_count

        for track_idx in final_unmatched_tracks:
            self.tracks[track_idx].misses += 1

        new_tracks = []
        for det_idx in final_unmatched_dets:
            det = detections[det_idx]
            box = (det.x1, det.y1, det.x2, det.y2)
            new_tracks.append(Track(self._next_id, box, det.label, det.confidence, 1, 0, 1 >= self.min_hits, self._frame_count, box, self._frame_count))
            self._next_id += 1

        self.tracks = [t for t in self.tracks if t.misses <= self.max_age] + new_tracks
        return [track for track in self.tracks if track.confirmed]

    def alive_track_ids(self) -> set[int]:
        """
        IDs of every track this Tracker is still holding internally, confirmed
        or not, as long as misses <= max_age. Wider than update()'s return value
        (which is confirmed-only). Callers keeping their own per-track_id state
        (e.g. entry/exit crossing state) can diff against this set each frame to
        remove IDs for tracks that have expired to avoid memory leakage for long 
        running streams.
        """
        return {track.track_id for track in self.tracks}

    def dwell_frames(self, track: Track) -> int:
        """How many frames since this track was first seen. 
        Converting to seconds is done by caller: dwell_frames(track) / fps"""
        return self._frame_count - track.first_seen_frame

    def stationary_frames(self, track: Track) -> int:
        """
        How many frames since this tracks position last relocated (moved further
        than stationary_threshold from its anchor). Caller converts to seconds.
        """
        return self._frame_count - track.stationary_since_frame


    def has_moved(self, track: Track) -> bool:
        """
        True if this track has relocated at least ocne since it was first seen.
        Can't distinguish always-static object from real person who was standing still
        from the first observed frame.
        """
        return track.stationary_since_frame != track.first_seen_frame


    


def _iou(boxA: tuple[float, float, float, float], boxB: tuple[float, float, float, float]) -> float:
    """Calculate IoU between a (track, detection) pair."""
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union_area = boxA_area + boxB_area - inter
    if union_area == 0:
        return 0.0

    return inter / union_area


def _centroid_dist(boxA: tuple[float, float, float, float], boxB: tuple[float, float, float, float]) -> float:
    """Calculate Euclidean distance between centroids of a (track, detection) pair."""
    cA = ((boxA[0] + boxA[2]) / 2.0, (boxA[1] + boxA[3]) / 2.0)
    cB = ((boxB[0] + boxB[2]) / 2.0, (boxB[1] + boxB[3]) / 2.0)

    return math.sqrt((cA[0] - cB[0]) ** 2 + (cA[1] - cB[1]) ** 2)


def _build_score_matrix(
    tracks: list[Track],
    detections: list[Detection],
    score_fn: Callable[[tuple[float, float, float, float], tuple[float, float, float, float]], float],
) -> list[list[float]]:
    """Build score matrix for matching (track, detection) pairs."""
    matrix = []
    for i in range(len(tracks)):
        row = []
        for j in range(len(detections)):
            det_box = (detections[j].x1, detections[j].y1, detections[j].x2, detections[j].y2)
            row.append(score_fn(tracks[i].box, det_box))
        matrix.append(row)
    return matrix


def _greedy_match(score_matrix: list[list[float]], threshold: float, higher_better: bool, num_cols: int) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """
    Greedy match the highest-scoring pair at a time.
    Returns the matched pairs, unmatched track (row) indexes, unmatched detection (col) indexes.

    num_cols must be passed explicitly (not inferred from score_matrix's shape): when there
    are zero tracks, score_matrix has zero rows, and an empty nested list can't tell you how
    many columns it "would have had", so the caller must pass that info.
    """
    triples = []
    num_rows = len(score_matrix)
    for r in range(num_rows):
        for c in range(num_cols):
            score = score_matrix[r][c]
            passes = score >= threshold if higher_better else score <= threshold
            if passes:
                triples.append((score, r, c))

    triples.sort(key=lambda t: t[0], reverse=higher_better)

    used_rows, used_cols, matched = set(), set(), []
    for score, r, c in triples:
        if r in used_rows or c in used_cols:
            continue
        matched.append((r, c))
        used_rows.add(r)
        used_cols.add(c)

    unmatched_rows = [r for r in range(num_rows) if r not in used_rows]
    unmatched_cols = [c for c in range(num_cols) if c not in used_cols]

    return matched, unmatched_rows, unmatched_cols