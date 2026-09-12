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


class FakeMovingThenStillDetector:
    """Moves once early, then holds still."""
    def __init__(self):
        self.frame_num = 0

    def detect(self, frame):
        self.frame_num += 1
        # relocate once on frame 1->2 then never again (e.g move once)
        box = (220, 100, 280, 300) if self.frame_num == 1 else (250, 100, 310, 300)
        det = Detection(label="person", confidence=0.9, x1=box[0], y1=box[1], x2=box[2], y2=box[3])
        return [det], 0.01


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


def test_loitering_alert_fires_after_stationary_time():
    """Loitering alert test through websocket route."""
    fake_redis = fakeredis.FakeRedis(decode_responses=True)
    shared_alert_manager = AlertManager(fake_redis, cooldown_seconds=30)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    num_frames = 310

    main_module.app.dependency_overrides[deps.get_alert_manager] = lambda: shared_alert_manager
    main_module.app.dependency_overrides[deps.get_detector] = lambda: FakeMovingThenStillDetector()
    main_module.app.dependency_overrides[deps.get_frame_source] = lambda: FakeFrameSource([frame] * num_frames)

    try:
        with TestClient(main_module.app) as test_client:
            with test_client.websocket_connect("/ws/detections") as ws:
                loitering_fired = False
                for _ in range(num_frames):
                    data = ws.receive_json()
                    for alert in data["alerts"]:
                        if alert["alert_type"] == "loitering":
                            loitering_fired = True
                assert loitering_fired
    finally:
        main_module.app.dependency_overrides.clear()


class FakeApproachingThenInsideDetector:
    """Static outside the zone for 3 frames (lets the track confirm before
    it ever touches the zone), then steps inward 40px/frame (< centroid_max_dist
    of 56px for a 640x480 frame, so the track ID is preserved) until it's
    inside. Zone for 640x480 is x1=213 - x2=426."""
    def __init__(self):
        self.frame_num = 0

    def detect(self, frame):
        self.frame_num += 1
        if self.frame_num <= 3:
            x1 = 470  # outside: x1 > zone x2=426
        else:
            x1 = 470 - 40 * (self.frame_num - 3)  # steps to 430, 390, 350 etc
        det = Detection(label="person", confidence=0.9, x1=x1, y1=100, x2=x1 + 60, y2=300)
        return [det], 0.01


def test_entry_event_and_occupancy_when_track_walks_into_zone():
    fake_redis = fakeredis.FakeRedis(decode_responses=True)
    shared_alert_manager = AlertManager(fake_redis, cooldown_seconds=30)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    num_frames = 7

    main_module.app.dependency_overrides[deps.get_alert_manager] = lambda: shared_alert_manager
    main_module.app.dependency_overrides[deps.get_detector] = lambda: FakeApproachingThenInsideDetector()
    main_module.app.dependency_overrides[deps.get_frame_source] = lambda: FakeFrameSource([frame] * num_frames)

    try:
        with TestClient(main_module.app) as test_client:
            with test_client.websocket_connect("/ws/detections") as ws:
                entry_events = []
                last_occupancy = None
                for _ in range(num_frames):
                    data = ws.receive_json()
                    entry_events += [e for e in data["crossing_events"] if e["event"] == "entry"]
                    last_occupancy = data["occupancy"]
                assert len(entry_events) == 1
                assert last_occupancy == {"entries": 1, "exits": 0}
    finally:
        main_module.app.dependency_overrides.clear()


class FakeInsideThenLeavingDetector:
    """Static inside the zone for 3 frames, then steps outward 40px/frame
    until it's fully outside."""
    def __init__(self):
        self.frame_num = 0

    def detect(self, frame):
        self.frame_num += 1
        if self.frame_num <= 3:
            x1 = 250  # inside: overlaps zone x1=213..x2=426
        else:
            x1 = 250 + 40 * (self.frame_num - 3)  # steps to 290, 330, 370, 410, 450 (outside)
        det = Detection(label="person", confidence=0.9, x1=x1, y1=100, x2=x1 + 60, y2=300)
        return [det], 0.01


def test_exit_event_and_occupancy_when_track_walks_out_of_zone():
    fake_redis = fakeredis.FakeRedis(decode_responses=True)
    shared_alert_manager = AlertManager(fake_redis, cooldown_seconds=30)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    num_frames = 8

    main_module.app.dependency_overrides[deps.get_alert_manager] = lambda: shared_alert_manager
    main_module.app.dependency_overrides[deps.get_detector] = lambda: FakeInsideThenLeavingDetector()
    main_module.app.dependency_overrides[deps.get_frame_source] = lambda: FakeFrameSource([frame] * num_frames)

    try:
        with TestClient(main_module.app) as test_client:
            with test_client.websocket_connect("/ws/detections") as ws:
                exit_events = []
                last_occupancy = None
                for _ in range(num_frames):
                    data = ws.receive_json()
                    exit_events += [e for e in data["crossing_events"] if e["event"] == "exit"]
                    last_occupancy = data["occupancy"]
                assert len(exit_events) == 1
                assert last_occupancy == {"entries": 0, "exits": 1}
    finally:
        main_module.app.dependency_overrides.clear()


def test_loitering_alert_doesnt_fire_for_false_positive():
    """A detection that's always at same position (the sign example)
    should never fire a loitering alert."""
    fake_redis = fakeredis.FakeRedis(decode_responses=True)
    shared_alert_manager = AlertManager(fake_redis, cooldown_seconds=30)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    num_frames = 310
 
    main_module.app.dependency_overrides[deps.get_alert_manager] = lambda: shared_alert_manager
    main_module.app.dependency_overrides[deps.get_detector] = fake_get_detector 
    main_module.app.dependency_overrides[deps.get_frame_source] = lambda: FakeFrameSource([frame] * num_frames)

    try:
        with TestClient(main_module.app) as test_client:
            with test_client.websocket_connect("/ws/detections") as ws:
                loitering_fired = False
                for _ in range(num_frames):
                    data = ws.receive_json()
                    for alert in data["alerts"]:
                        if alert["alert_type"] == "loitering":
                            loitering_fired = True
                assert loitering_fired is False
    finally:
        main_module.app.dependency_overrides.clear()


class FakePackageThenGoneDetector:
    """
    Backpack inside zone long enouhg to be flagged 'left'
    then dissapears, long enough to exceed max_age and be
    flagged 'taken.
    """
    def __init__(self, left_frames):
        self.frame_num = 0
        self.left_frames = left_frames

    def detect(self, frame):
        self.frame_num += 1
        if self.frame_num <= self.left_frames:
            det = Detection(label="backpack", confidence=0.9, x1=250, y1=100, x2=310, y2=200)
            return [det], 0.01
        return [], 0.01


def test_package_left_then_taken():
    fake_redis = fakeredis.FakeRedis(decode_responses=True)
    shared_alert_manager = AlertManager(fake_redis, cooldown_seconds=30)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    left_frames = 151 # (151-1)/30 = 5.0s crosses PACKAGE_LEFT_SECONDS
    num_frames = left_frames + 10 # + empty frames to exceed max_age=8

    main_module.app.dependency_overrides[deps.get_alert_manager] = lambda: shared_alert_manager
    main_module.app.dependency_overrides[deps.get_detector] = lambda: FakePackageThenGoneDetector(left_frames)
    main_module.app.dependency_overrides[deps.get_frame_source] = lambda: FakeFrameSource([frame] * num_frames)

    try:
        with TestClient(main_module.app) as test_client:
            with test_client.websocket_connect("/ws/detections") as ws:
                left_events = []
                taken_events = []
                taken_alert_fired = False
                last_packages_count = None

                for _ in range(num_frames):
                    data = ws.receive_json()
                    left_events += [e for e in data["package_events"] if e["event"] == "left"]
                    taken_events += [e for e in data["package_events"] if e["event"] == "taken"]
                    for alert in data["alerts"]:
                        if alert["alert_type"] == "package_taken":
                            taken_alert_fired = True
                    last_packages_count = data["packages"]

                assert len(left_events) == 1
                assert left_events[0]["label"] == "backpack"
                assert len(taken_events) == 1
                assert taken_alert_fired
                assert last_packages_count == {"left": 1, "taken": 1}
    finally:
        main_module.app.dependency_overrides.clear()


def make_test_jpeg():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes()


def test_demo_detect_returns_json_detections(client):
    response = client.post(
        "/demo/detect",
        files={"file": ("test.jpg", make_test_jpeg(), "image/jpeg")},
        params={"format": "json"},
    )
    assert response.status_code == 200
    body = response.json()
    assert "detections" in body
    assert "timing" in body


def test_demo_detect_no_auth_required_when_api_key_unset(client, monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    response = client.post(
        "/demo/detect",
        files={"file": ("test.jpg", make_test_jpeg(), "image/jpeg")},
        params={"format": "json"},
    )
    assert response.status_code == 200


def test_demo_detect_rejects_missing_api_key_when_set(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "correct=secret")
    response = client.post(
        "/demo/detect",
        files={"file": ("test.jpg", make_test_jpeg(), "image/jpeg")},
        params={"format": "json"},
    )
    assert response.status_code == 401


def test_demo_detect_rejects_wrong_api_key(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "correct-secret")
    response = client.post(
        "/demo/detect",
        files={"file": ("test.jpg", make_test_jpeg(), "image/jpeg")},
        params={"format": "json"},
        headers={"x-api-key": "wrong-secret"},
    )
    assert response.status_code == 401


def test_demo_detect_accepts_correct_api_key(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "correct-secret")
    response = client.post(
        "/demo/detect",
        files={"file": ("test.jpg", make_test_jpeg(), "image/jpeg")},
        params={"format": "json"},
        headers={"x-api-key": "correct-secret"},
    )
    assert response.status_code == 200


class TrackingFrameSourceFactory:
    """
    Wrap FakeFrameSource construction so test can verify it was 
    actually called, proving that an unauthorized websocket connection
    never triggers get_frame_source() to open a video device.
    """
    def __init__(self, frames):
        self.frames = frames
        self.was_called = False

    def __call__(self):
        self.was_called = True
        return FakeFrameSource(self.frames)


def test_websocket_rejects_connection_no_api_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "correct-secret")
    fake_redis = fakeredis.FakeRedis(decode_responses=True)
    shared_alert_manager = AlertManager(fake_redis, cooldown_seconds=30)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    main_module.app.dependency_overrides[deps.get_alert_manager] = lambda: shared_alert_manager
    main_module.app.dependency_overrides[deps.get_detector] = lambda: FakeDetector()
    main_module.app.dependency_overrides[deps.get_frame_source] = lambda: FakeFrameSource([frame])

    try:
        with TestClient(main_module.app) as test_client:
            with pytest.raises(Exception):
                with test_client.websocket_connect("/ws/detections") as ws:
                    ws.receive_json()
    finally:
        main_module.app.dependency_overrides.clear()


def test_websocket_unauthorized_connection_never_opens_video_source(monkeypatch):
    monkeypatch.setenv("API_KEY", "correct-secret")
    fake_redis = fakeredis.FakeRedis(decode_responses=True)
    shared_alert_manager = AlertManager(fake_redis, cooldown_seconds=30)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    tracking_factory = TrackingFrameSourceFactory([frame])

    main_module.app.dependency_overrides[deps.get_alert_manager] = lambda: shared_alert_manager
    main_module.app.dependency_overrides[deps.get_detector] = lambda: FakeDetector()
    main_module.app.dependency_overrides[deps.get_frame_source] = tracking_factory

    try:
        with TestClient(main_module.app) as test_client:
            with pytest.raises(Exception):
                with test_client.websocket_connect("/ws/detections") as ws:
                    ws.receive_json()
        assert tracking_factory.was_called is False
    finally:
        main_module.app.dependency_overrides.clear()


def test_websocket_accepts_connection_with_correct_api_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "correct-secret")
    fake_redis = fakeredis.FakeRedis(decode_responses=True)
    shared_alert_manager = AlertManager(fake_redis, cooldown_seconds=30)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    main_module.app.dependency_overrides[deps.get_alert_manager] = lambda: shared_alert_manager
    main_module.app.dependency_overrides[deps.get_detector] = lambda: FakeDetector()
    main_module.app.dependency_overrides[deps.get_frame_source] = lambda: FakeFrameSource([frame])

    try:
        with TestClient(main_module.app) as test_client:
            with test_client.websocket_connect("/ws/detections?api_key=correct-secret") as ws:
                data = ws.receive_json()
                assert "detections" in data
    finally:
        main_module.app.dependency_overrides.clear()
        