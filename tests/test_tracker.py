from app.detector import Detection
from app.tracker import Tracker


def make_det(x1, y1, x2, y2, conf=0.9):
    return Detection(label="person", confidence=conf, x1=x1, y1=y1, x2=x2, y2=y2)


def test_track_not_confirmed_from_first_detection():
    tracker = Tracker(min_hits=3)
    confirmed = tracker.update([make_det(10, 10, 50, 90)])

    assert confirmed == []
    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].hits == 1
    assert tracker.tracks[0].confirmed is False


def test_track_confirms_after_min_hits():
    tracker = Tracker(min_hits=3)
    box = (10, 10, 50, 90)

    tracker.update([make_det(*box)])
    tracker.update([make_det(*box)])
    confirmed = tracker.update([make_det(*box)])

    assert len(confirmed) == 1
    assert confirmed[0].hits == 3


def test_track_survives_one_frame_gap():
    tracker = Tracker(min_hits=2, max_age=3)
    box = (100, 100, 150, 200)

    tracker.update([make_det(*box)])
    confirmed = tracker.update([make_det(*box)])
    assert len(confirmed) == 1
    original_id = confirmed[0].track_id

    # already confirmed, so a single miss keeps it in the confirmed list 
    confirmed = tracker.update([])
    assert len(confirmed) == 1
    assert confirmed[0].track_id == original_id

    confirmed = tracker.update([make_det(*box)])
    assert len(confirmed) == 1
    assert confirmed[0].track_id == original_id


def test_track_dies_after_max_age_consecutive_misses():
    tracker = Tracker(min_hits=2, max_age=2)
    box = (100, 100, 150, 200)

    tracker.update([make_det(*box)])
    tracker.update([make_det(*box)])

    tracker.update([])
    tracker.update([])
    tracker.update([])

    assert tracker.tracks == []


def test_separated_tracks_do_not_swap_ids():
    tracker = Tracker(min_hits=2)
    box_left = (0, 0, 40, 80)
    box_right = (500, 0, 540, 80)

    tracker.update([make_det(*box_left), make_det(*box_right)])
    confirmed = tracker.update([make_det(*box_left), make_det(*box_right)])
    assert len(confirmed) == 2

    id_left = next(t.track_id for t in confirmed if t.box[0] < 100)
    id_right = next(t.track_id for t in confirmed if t.box[0] > 100)

    box_left_moved = (5, 0, 45, 80)
    box_right_moved = (505, 0, 545, 80)
    confirmed = tracker.update([make_det(*box_left_moved), make_det(*box_right_moved)])

    new_id_left = next(t.track_id for t in confirmed if t.box[0] < 100)
    new_id_right = next(t.track_id for t in confirmed if t.box[0] > 100)

    assert new_id_left == id_left
    assert new_id_right == id_right


def test_track_carries_label_and_confidence_from_matched_detection():
    """Track.label/confidence should reflect the real detection, not a hardcoded placeholder."""
    tracker = Tracker(min_hits=1)
    confirmed = tracker.update([make_det(10, 10, 50, 90, conf=0.42)])

    assert confirmed[0].label == "person"
    assert confirmed[0].confidence == 0.42

    # confidence should refresh on each new match, not freeze at creation
    confirmed = tracker.update([make_det(10, 10, 50, 90, conf=0.77)])
    assert confirmed[0].confidence == 0.77