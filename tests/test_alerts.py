import fakeredis
import pytest

from app.alerts import AlertManager, Zone


@pytest.fixture
def zone():
    return Zone(name="front_door", x1=100, y1=100, x2=300, y2=300)


def test_overlaps_when_boxes_intersect(zone):
    assert zone.overlaps_box(150, 150, 250, 250) is True


def test_overlaps_when_box_fully_inside_zone(zone):
    assert zone.overlaps_box(120, 120, 200, 200) is True


def test_overlaps_when_zone_fully_inside_box(zone):
    assert zone.overlaps_box(0, 0, 400, 400) is True


def test_no_overlap_when_box_entirely_left(zone):
    assert zone.overlaps_box(0, 150, 50, 250) is False


def test_no_overlap_when_box_entirely_right(zone):
    assert zone.overlaps_box(350, 150, 400, 250) is False


def test_no_overlap_when_box_entirely_above(zone):
    assert zone.overlaps_box(150, 0, 250, 50) is False


def test_no_overlap_when_box_entirely_below(zone):
    assert zone.overlaps_box(150, 350, 250, 400) is False


def test_overlap_when_edges_touch(zone):
    # box's right edge exactly meets the zone's left edge
    assert zone.overlaps_box(0, 150, 100, 250) is True


@pytest.fixture
def alert_manager():
    fake_redis = fakeredis.FakeRedis(decode_responses=True)
    return AlertManager(fake_redis, cooldown_seconds=30)


def test_first_alert_fires(alert_manager):
    alert = alert_manager.raise_if_new("front_door", "person", 0.9)
    assert alert is not None
    assert alert.zone_name == "front_door"
    assert alert.label == "person"


def test_second_alert_suppressed_within_cooldown(alert_manager):
    alert_manager.raise_if_new("front_door", "person", 0.9)
    second = alert_manager.raise_if_new("front_door", "person", 0.8)
    assert second is None


def test_different_zone_not_suppressed(alert_manager):
    alert_manager.raise_if_new("front_door", "person", 0.9)
    other = alert_manager.raise_if_new("back_gate", "person", 0.9)
    assert other is not None


def test_different_label_not_suppressed(alert_manager):
    alert_manager.raise_if_new("front_door", "person", 0.9)
    other = alert_manager.raise_if_new("front_door", "dog", 0.9)
    assert other is not None


def test_different_alert_type_not_suppressed(alert_manager):
    alert_manager.raise_if_new("front_door", "person", 0.9, "zone_entry")
    loiter = alert_manager.raise_if_new("front_door", "person", 0.9, "loitering")
    assert loiter is not None
    assert loiter.alert_type == "loitering"


def test_alert_type_defaults_to_zone_entry(alert_manager):
    alert = alert_manager.raise_if_new("front_door", "person", 0.9)
    assert alert.alert_type == "zone_entry"


def test_recent_alerts_returns_most_recent_first(alert_manager):
    alert_manager.raise_if_new("zone_a", "person", 0.9)
    alert_manager.raise_if_new("zone_b", "person", 0.9)
    recent = alert_manager.recent_alerts(limit=10)
    assert recent[0]["zone_name"] == "zone_b"
    assert recent[1]["zone_name"] == "zone_a"


def test_recent_alerts_respects_limit(alert_manager):
    for i in range(5):
        alert_manager.raise_if_new(f"zone_{i}", "person", 0.9)
    assert len(alert_manager.recent_alerts(limit=2)) == 2


def test_recent_alerts_zero_limit_returns_empty(alert_manager):
    alert_manager.raise_if_new("zone_a", "person", 0.9)
    assert alert_manager.recent_alerts(limit=0) == []
