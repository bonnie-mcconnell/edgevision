# Redis SET NX EX gives us an expiring "already alerted?" check for free so no cleanup job needed.

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import redis


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


@dataclass
class Alert:
    zone_name: str
    label: str  # e.g "person"
    confidence: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "zone_name": self.zone_name,
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "timestamp": self.timestamp,
        }


class AlertManager:
    def __init__(self, redis_client: redis.Redis, cooldown_seconds: int = 30):
        self.redis = redis_client
        self.cooldown_seconds = cooldown_seconds

    def _dedup_key(self, zone_name: str, label: str) -> str:
        return f"alert:cooldown:{zone_name}:{label}"

    def should_fire(self, zone_name: str, label: str) -> bool:
        """True if this zone/label hasn't fired within the cooldown window."""
        key = self._dedup_key(zone_name, label)
        was_set = self.redis.set(key, "1", nx=True, ex=self.cooldown_seconds)
        return bool(was_set)

    def raise_if_new(self, zone_name: str, label: str, confidence: float) -> Alert | None:
        """Logs and returns an Alert if not in cooldown, otherwise returns None."""
        if self.should_fire(zone_name, label):
            alert = Alert(zone_name=zone_name, label=label, confidence=confidence)
            self.redis.lpush("alert:log", json.dumps(alert.to_dict()))
            self.redis.ltrim("alert:log", 0, 199)  # keep the log bounded to 200 alerts
            return alert
        return None

    def recent_alerts(self, limit: int = 20) -> list[dict]:
        if limit <= 0:
            return []
        raw = self.redis.lrange("alert:log", 0, limit - 1) # type: ignore[misc]
        return [json.loads(entry) for entry in raw]  
