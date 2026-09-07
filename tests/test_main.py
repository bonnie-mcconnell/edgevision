import fakeredis
import pytest
import cv2
from fastapi.testclient import TestClient
import numpy as np

import app.dependencies as deps
import app.main as main_module
from app.alerts import AlertManager
from app.detector import Detection


class FakeDetector:
    def detect(self, frame):
        detection = Detection(label="person", confidence=0.9, x1=250, y1=100, x2=350, y2=300)
        return [detection], 0.01


class FakeFrameSource:
    def __init__(self, frames):
        self._frames = frames # list of numpy arrays
        self._index = 0

    def isOpened(self):
        return True

    def read(self):
        if self._index < len(self._frames):
            frame = self._frames[self._index]
            self._index += 1
            return True, frame
        
        return False, None

    def release(self):
        pass

    def get(self, prop_id):
        # need this to scale centroid_max_dist and compute dwell_seconds
        h, w = self._frames[0].shape[:2] if self._frames else (480, 640)
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return w
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return h
        if prop_id == cv2.CAP_PROP_FPS:
            return 30.0
        return 0
        

def fake_get_detector():
    return FakeDetector()


def fake_get_redis_client():
    return fakeredis.FakeRedis(decode_responses=True)


def fake_get_frame_source():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Tracking requires min_hits consecutive matched frames before a
    # track confirms and can alert (default min_hits=3)
    return FakeFrameSource([frame, frame, frame])

@pytest.fixture
def client():
    fake_redis = fakeredis.FakeRedis(decode_responses=True)
    shared_alert_manager = AlertManager(fake_redis, cooldown_seconds=30)

    main_module.get_detector = fake_get_detector
    main_module.get_redis_client = fake_get_redis_client

    main_module.app.dependency_overrides[deps.get_alert_manager] = (
        lambda: shared_alert_manager
    )
    main_module.app.dependency_overrides[deps.get_detector] = fake_get_detector
    main_module.app.dependency_overrides[deps.get_frame_source] = fake_get_frame_source

    with TestClient(main_module.app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_recent_alerts_empty(client):
    response = client.get("/alerts/recent")
    assert len(response.json()["alerts"]) == 0


def test_alert_full_pipeline(client):
    with client.websocket_connect("/ws/detections") as ws:
        # the same detection needs to repeat for min_hits frames before its
        # track confirms and the alert can fire, so read every frame the fake
        # source produces and check across all of them, not just the first
        alert_fired = False
        for _ in range(3):
            data = ws.receive_json()
            if data["alerts"]:
                alert_fired = True
        assert alert_fired

    response = client.get("/alerts/recent")
    assert len(response.json()["alerts"]) == 1