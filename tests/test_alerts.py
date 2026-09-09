import fakeredis
import pytest

from app.alerts import AlertManager, EntryExitCounter, Zone
from app.tracker import Track


OUTSIDE_BOX = (0, 150, 50, 250)   #left of zone (x1=100..x2=300)
INSIDE_BOX = (150, 150, 250, 250)


def make_track(track_id, box, first_seen_frame=1):
    return Track(
        track_id=track_id, box=box, label="person", confidence=0.9,
        hits=3, misses=0, confirmed=True, first_seen_frame=first_seen_frame,
        anchor_box=box, stationary_since_frame=first_seen_frame,
    )


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


def test_first_observation_does_not_fire_event(zone):
    counter = EntryExitCounter()
    track = make_track(1, INSIDE_BOX)

    events = counter.update([track], zone, alive_ids={1})

    assert events == []
    assert counter.entries == 0
    assert counter.exits == 0


def test_entry_fires_when_track_moves_into_zone(zone):
    counter = EntryExitCounter()
    counter.update([make_track(1, OUTSIDE_BOX)], zone, alive_ids={1})

    events = counter.update([make_track(1, INSIDE_BOX)], zone, alive_ids={1})

    assert events == [{"track_id": 1, "event": "entry"}]
    assert counter.entries == 1
    assert counter.exits == 0


def test_exit_fires_when_track_leaves_zone(zone):
    counter = EntryExitCounter()
    counter.update([make_track(1, INSIDE_BOX)], zone, alive_ids={1}) # seed as inside

    events = counter.update([make_track(1, OUTSIDE_BOX)], zone, alive_ids={1})

    assert events == [{"track_id": 1, "event": "exit"}]
    assert counter.exits == 1
    assert counter.entries == 0


def test_no_event_while_track_stays_inside(zone):
    counter = EntryExitCounter()
    counter.update([make_track(1, INSIDE_BOX)], zone, alive_ids={1})

    events = counter.update([make_track(1, INSIDE_BOX)], zone, alive_ids={1})

    assert events == []
    assert counter.entries == 0
    assert counter.exits == 0


def test_stale_track_state_pruned_when_no_longer_alive(zone):
    counter = EntryExitCounter()
    counter.update([make_track(1, INSIDE_BOX)], zone, alive_ids={1})

    # track 1 has died (max_age exceeded), alive_ids no longer includes it
    counter.update([], zone, alive_ids=set())
    assert counter._inside == {}

    # new track reusing distinct id shouldn't inherit old state
    events = counter.update([make_track(2, INSIDE_BOX)], zone, alive_ids={2})
    assert events == []  # treated as first observation correctly


def test_entries_and_exits_accumulate_across_multiple_tracks(zone):
    counter = EntryExitCounter()
    counter.update([make_track(1, OUTSIDE_BOX), make_track(2, INSIDE_BOX)], zone, alive_ids={1, 2})

    events = counter.update([make_track(1, INSIDE_BOX), make_track(2, OUTSIDE_BOX)], zone, alive_ids={1, 2})

    assert {"track_id": 1, "event": "entry"} in events
    assert {"track_id": 2, "event": "exit"} in events
    assert counter.entries == 1
    assert counter.exits == 1