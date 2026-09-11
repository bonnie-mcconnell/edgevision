# Redis SET NX EX gives us an expiring "already alerted?" check for free so no cleanup job needed.

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import redis

from app.detector import PACKAGE_CLASSES
from app.tracker import Track, Tracker


@dataclass(frozen=True)
class Zone:
    """A rectangular region in the frame, in pixel coordinates (x1, y1, x2, y2)."""
    name: str
    x1: int
    y1: int
    x2: int
    y2: int

    def overlaps_box(self, box_x1: float, box_y1: float, box_x2: float, box_y2: float) -> bool:
        """True if a detection box overlaps this zone at all (not just its center)."""
        return not (
            box_x2 < self.x1 or box_x1 > self.x2 or
            box_y2 < self.y1 or box_y1 > self.y2
        )


def default_zone_for_resolution(width: float, height: float, name: str = "front_door") -> Zone:
    """
    Returns Zone bounded to the middle third of the frame, scaled to the
    actual images resolution. TODO: real deployments would want a per camera
    configured zone drawn by a user in a setup UI.
    """
    return Zone(name=name, x1=int(width/3), y1=0, x2=int(2 * width/3), y2=int(height))


# TODO: to be tuned
LOITERING_SECONDS = 10.0
PACKAGE_LEFT_SECONDS = 5.0


class PackageMonitor:
    """
    Tracks package-class objects through left -> taken.
    "left" fires once a package track has sat stationary in the zone
    for PACKAGE_LEFT_SECONDS, "taken" fires when a previously flagged 
    package's track is no longer alive.

    Doesn't gate on has_moved() as it could only be picked up once it's
    been left, but alos means that a permanently static misdetection looks
    identical to a real left package. Would need a longer PACKAGE_LEFT_SECONDS
    or class-confidence stability check across frames to fix.
    """
    def __init__(self, left_seconds: float = PACKAGE_LEFT_SECONDS):
        # track_id: (label, confidence) when left fires
        self._flagged_left: dict[int, tuple[str, float]] = {}
        self.left_seconds = left_seconds
        self.left_count = 0
        self.taken_count = 0

    def update(self, tracks: list[Track], zone: Zone, tracker: Tracker, fps: float, alive_ids: set[int]) -> list[dict]:
        """
        Call once per frame with the tracker's confirmed tracks. Returns list of
        {"track_id": int, "event": "left"|"taken", "label": str, "confidence": float}
        for transitions detected this frame. "taken" events carry the label/confidence
        remembered from when "left" fired because the track no longer exists at that point.
        """
        events = []

        taken_ids = set(self._flagged_left) - alive_ids
        for tid in taken_ids:
            label, confidence = self._flagged_left.pop(tid)
            self.taken_count += 1
            events.append({"track_id": tid, "event": "taken", "label": label, "confidence": confidence})

        for track in tracks:
            if track.label not in PACKAGE_CLASSES:
                continue 
            if track.track_id in self._flagged_left:
                continue
            if not zone.overlaps_box(*track.box):
                continue
            stationary_seconds = tracker.stationary_frames(track) / fps
            if stationary_seconds >= self.left_seconds:
                self._flagged_left[track.track_id] = (track.label, track.confidence)
                self.left_count += 1
                events.append({"track_id": track.track_id, "event": "left", "label": track.label, "confidence": track.confidence})

        return events


class EntryExitCounter:
    """
    Turns per-frame in-zone checks into entry/exit events by diffing against
    each track's inside/outside state from the previous frame.

    Limitations: 
    - A track's first observation only seeds its state, it never fires an
      event. A track that's already inside the zone the first frame it's 
      confirmed is indistinguishable from one that just walked in, so it's not
      counted as an entry.
    - Only confirmed tracks are counted so a fast entry can be fully inside the zone
      before it's first observed here and never register as an entry event, only
      as an exit.
    - If a track dies while still inside the zone (past max_age, leaves frame) no exit 
      event fires. entries - exits as live occupancy count overcounts in this case.
    """
    def __init__(self):
        self._inside: dict[int, bool] = {}
        self.entries = 0
        self.exits = 0

    def update(self, tracks: list[Track], zone: Zone, alive_ids: set[int]) -> list[dict]:
        """
        Call once per frame with the tracker's confirmed tracks.
        Returns list of {"track_id": int, "event": "entry"|"exit"} for 
        transitions found this frame.
        """
        self._inside = {tid: was_inside for tid, was_inside in self._inside.items() if tid in alive_ids}

        events = []
        for track in tracks:
            is_inside = zone.overlaps_box(*track.box)
            was_inside = self._inside.get(track.track_id)

            if was_inside is None:
                self._inside[track.track_id] = is_inside
                continue

            if is_inside and not was_inside:
                self.entries += 1
                events.append({"track_id": track.track_id, "event": "entry"})
            elif was_inside and not is_inside:
                self.exits += 1
                events.append({"track_id": track.track_id, "event": "exit"})

            self._inside[track.track_id] = is_inside 

        return events


@dataclass
class Alert:
    zone_name: str
    label: str  # e.g "person"
    confidence: float
    alert_type: str = "zone_entry"
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "zone_name": self.zone_name,
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "alert_type": self.alert_type,
            "timestamp": self.timestamp,
        }


class AlertManager:
    def __init__(self, redis_client: redis.Redis, cooldown_seconds: int = 30):
        self.redis = redis_client
        self.cooldown_seconds = cooldown_seconds

    def _dedup_key(self, zone_name: str, label: str, alert_type: str) -> str:
        return f"alert:cooldown:{alert_type}:{zone_name}:{label}"

    def should_fire(self, zone_name: str, label: str, alert_type: str = "zone_entry") -> bool:
        """True if this zone/label hasn't fired within the cooldown window."""
        key = self._dedup_key(zone_name, label, alert_type)
        was_set = self.redis.set(key, "1", nx=True, ex=self.cooldown_seconds)
        return bool(was_set)

    def raise_if_new(self, zone_name: str, label: str, confidence: float, alert_type: str = "zone_entry") -> Alert | None:
        """Logs and returns an Alert if not in cooldown, otherwise returns None."""
        if self.should_fire(zone_name, label, alert_type):
            alert = Alert(zone_name=zone_name, label=label, confidence=confidence, alert_type=alert_type)
            self.redis.lpush("alert:log", json.dumps(alert.to_dict()))
            self.redis.ltrim("alert:log", 0, 199)  # keep the log bounded to 200 alerts
            return alert
        return None

    def recent_alerts(self, limit: int = 20) -> list[dict]:
        if limit <= 0:
            return []
        raw = self.redis.lrange("alert:log", 0, limit - 1) # type: ignore[misc]
        return [json.loads(entry) for entry in raw]  
