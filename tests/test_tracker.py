import numpy as np

from app.detector import Detection
from app.tracker import Tracker, _greedy_match, _hungarian_match


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


def test_stationary_frames_increases_for_stationary_person():
    tracker = Tracker(min_hits=1, stationary_threshold=20.0)

    boxes = [
        (100, 100, 150, 200), (101, 100, 151, 200), (100, 101, 150, 201),
        (102, 100, 152, 200), (100, 100, 150, 200),
    ]
    confirmed = None
    for box in boxes:
        confirmed = tracker.update([make_det(*box)])

    assert confirmed is not None
    assert tracker.stationary_frames(confirmed[0]) == 4 


def test_stationary_frames_resets_on_relocation():
    tracker = Tracker(min_hits=1, stationary_threshold=20.0)
    box_here = (100, 100, 150, 200)
    # centroid moved 40px (stationary_threshold < 40 < centroid_max_dist)
    box_relocated = (140, 100, 190, 200)

    tracker.update([make_det(*box_here)])
    tracker.update([make_det(*box_here)])
    tracker.update([make_det(*box_here)])
    confirmed = tracker.update([make_det(*box_relocated)])

    assert len(confirmed) == 1 # not a new track
    assert tracker.stationary_frames(confirmed[0]) == 0


def test_stationary_frames_continues_accumulating_under_threshold():
    """Test jitter e.g a move thats real but under stationary_threshold."""
    tracker = Tracker(min_hits=1, stationary_threshold=20.0)
    box_a = (100, 100, 150, 200)
    box_b = (105, 100, 155, 200) # centroid moved 5px

    tracker.update([make_det(*box_a)])
    tracker.update([make_det(*box_a)])
    confirmed = tracker.update([make_det(*box_b)])

    assert tracker.stationary_frames(confirmed[0]) == 2


def test_has_moved_false_for_unmoving_track():
    tracker = Tracker(min_hits=1, stationary_threshold=20.0)
    box = (100, 100, 150, 200)

    confirmed = None
    for _ in range(10):
        confirmed = tracker.update([make_det(*box)])

    assert confirmed is not None
    assert tracker.has_moved(confirmed[0]) is False
    assert tracker.stationary_frames(confirmed[0]) == 9


def test_has_moved_true_after_reloaction():
    tracker = Tracker(min_hits=1, stationary_threshold=20.0)
    box_start = (100, 100, 150, 200)
    box_after = (140, 100, 190, 200) # centroid moved 40px

    tracker.update([make_det(*box_start)])
    confirmed = tracker.update([make_det(*box_after)])

    assert len(confirmed) == 1
    assert tracker.has_moved(confirmed[0]) is True


def test_alive_track_ids_includes_unconfirmed_track():
    tracker = Tracker(min_hits=3)
    tracker.update([make_det(10, 10, 50, 90)])

    assert len(tracker.alive_track_ids()) == 1


def test_alive_track_ids_includes_confirmed_track_during_miss_gap():
    tracker = Tracker(min_hits=1, max_age=3)
    box = (100, 100, 150, 200)

    confirmed = tracker.update([make_det(*box)])
    track_id = confirmed[0].track_id

    tracker.update([])  # one missed frame, still within max_age

    assert track_id in tracker.alive_track_ids()


def test_alive_track_ids_excludes_track_after_it_dies():
    tracker = Tracker(min_hits=1, max_age=2)
    box = (100, 100, 150, 200)

    confirmed = tracker.update([make_det(*box)])
    track_id = confirmed[0].track_id

    tracker.update([])
    tracker.update([])
    tracker.update([])  # exceeds max_age, track should be gone

    assert track_id not in tracker.alive_track_ids()
    assert tracker.alive_track_ids() == set()


def test_hungarian_beats_greedy():
    """
    Greedy matching takes the single best pair at a time, e.g
    takes (A-Det1, 0.60), then remaining (B-Det2, 0.05) which
    is left unmatched. Hungarian finds optimal global assignment
    (A-Det2 + B-Det1 = 1.05) for all tracks.
    """
    iou_matrix = [
        [0.60, 0.50],
        [0.55, 0.05],
    ]

    greedy_matched, _, _ = _greedy_match(iou_matrix, threshold=0.3, higher_better=True, num_cols=2)
    hungarian_matched, hung_unmatched_rows, hung_unmatched_cols = _hungarian_match(
        iou_matrix, threshold=0.3, higher_better=True, num_cols=2
    )

    assert greedy_matched == [(0, 0)] # leaves track 1 unmatched
    assert set(hungarian_matched) == {(0, 1), (1, 0)} # matches both
    assert hung_unmatched_rows == []
    assert hung_unmatched_cols == []

    greedy_total = sum(iou_matrix[r][c] for r, c in greedy_matched)
    hungarian_total = sum(iou_matrix[r][c] for r, c in hungarian_matched)
    assert hungarian_total > greedy_total


def test_hungarian_empty_matrix():
    assert _hungarian_match([], threshold=0.3, higher_better=True, num_cols=0) == ([], [], [])


def test_hungarian_zero_tracks():
    matched, unmatched_rows, unmatched_cols = _hungarian_match(
        [[], []], threshold=0.3, higher_better=True, num_cols=0
    )
    assert matched == []
    assert unmatched_rows == [0, 1]
    assert unmatched_cols == []


def test_hungarian_rectangular():
    """3 tracks & 2 detections makes a rectangular matrix, one track
    must be unmatched. (0, 1) + (2, 0) = 1.7 is the optimal solution."""
    matrix = [
        [0.5, 0.9],
        [0.2, 0.6],
        [0.8, 0.3],
    ]
    matched, unmatched_rows, unmatched_cols = _hungarian_match(matrix, threshold=0.1, higher_better=True, num_cols=2)

    assert set(matched) == {(0, 1), (2, 0)}
    assert unmatched_rows == [1]
    assert unmatched_cols == []


def test_hungarian_threshold_rejects_optimal_pairing():
    """
    Hungarian finds optimal assignment, threshold checks afterwards
    and rejects bad matches.
    """
    all_bad = [[0.05, 0.02], [0.01, 0.03]]
    matched, unmatched_rows, unmatched_cols = _hungarian_match(all_bad, threshold=0.3, higher_better=True, num_cols=2)

    assert matched == []
    assert unmatched_rows == [0, 1]
    assert unmatched_cols == [0, 1]


def test_hungarian_lower_better():
    """higher_better=False for centroid distance. Smaller
    values are better matches, matrix used as cost."""
    distance_matrix = [
        [10.0, 80.0],
        [70.0, 15.0],
    ]

    matched, unmatched_rows, unmatched_cols = _hungarian_match(
        distance_matrix, threshold=20.0, higher_better=False, num_cols=2
    )
    assert set(matched) == {(0, 0), (1, 1)}
    assert unmatched_rows == []
    assert unmatched_cols == []


def test_hungarian_greedy_agrees():
    """
    When only one correct pairing exists both algorithms agree.
    """
    matrix = [
        [0.9, 0.05],
        [0.05, 0.85],
    ]
    greedy_matched, _, _ = _greedy_match(matrix, threshold=0.3, higher_better=True, num_cols=2)
    hungarian_matched, _, _ = _hungarian_match(matrix, threshold=0.3, higher_better=True, num_cols=2)

    assert set(greedy_matched) == set(hungarian_matched) == {(0, 0), (1, 1)}


def test_tracker_defaults_hungarian():
    tracker = Tracker()
    assert tracker.match_fn is _hungarian_match


def test_tracker_custom_match_fn():
    tracker = Tracker(match_fn=_greedy_match)
    assert tracker.match_fn is _greedy_match


def make_frame_with_box(color, box, size=(200, 300)):
    frame = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    x1, y1, x2, y2 = box
    frame[y1:y2, x1:x2] = color
    return frame

RED = (30, 30, 200)
BLUE = (200, 30, 30)
ORIGINAL_BOX = (20, 20, 80, 120)
FAR_BOX = (200, 20, 260, 120) # centroid 180px from original box


def test_revivial_reunites_dead_track():
    # track that died due to max_age that reappears
    # should get its original identity back if its 
    # appearance matches
    tracker = Tracker(min_hits=1, max_age=2, centroid_max_dist=50.0,
                      revival_max_age=20, appearance_threshold=0.5)
    frame = make_frame_with_box(RED, ORIGINAL_BOX)
    original_det = Detection(label="person", confidence=0.9, x1=20, y1=20, x2=80, y2=120)
    confirmed = tracker.update([original_det], frame=frame)
    original_id = confirmed[0].track_id

    # occlusion, track dies and appearance is saved
    for _ in range(3):
        tracker.update([], frame=frame)

    reappear_frame = make_frame_with_box(RED, FAR_BOX)
    reappear_det = Detection(label="person", confidence=0.9, x1=200, y1=20, x2=260, y2=120)
    confirmed = tracker.update([reappear_det], frame=reappear_frame)

    assert confirmed[0].track_id == original_id



def test_no_revival_appearance_doesnt_match():
    tracker = Tracker(min_hits=1, max_age=2, centroid_max_dist=50.0,
                       revival_max_age=20, appearance_threshold=0.5)

    frame = make_frame_with_box(RED, ORIGINAL_BOX)
    original_det = Detection(label="person", confidence=0.9, x1=20, y1=20, x2=80, y2=120)
    confirmed = tracker.update([original_det], frame=frame)
    original_id = confirmed[0].track_id

    for _ in range(3):
        tracker.update([],frame=frame)

    reappear_frame = make_frame_with_box(BLUE, FAR_BOX)
    reappear_det = Detection(label="person", confidence=0.9, x1=200, y1=20, x2=260, y2=120)
    confirmed = tracker.update([reappear_det], frame=reappear_frame)

    assert confirmed[0].track_id != original_id


def test_no_frame_param():
    # preserve backward compatibility, passing no frame = no revival
    tracker = Tracker(min_hits=1, max_age=2, centroid_max_dist=50.0)

    original_det = Detection(label="person", confidence=0.9, x1=20, y1=20, x2=80, y2=120)
    confirmed = tracker.update([original_det])  # no frame
    original_id = confirmed[0].track_id

    for _ in range(3):
        tracker.update([])

    far_det = Detection(label="person", confidence=0.9, x1=200, y1=20, x2=260, y2=120)
    confirmed = tracker.update([far_det])

    assert confirmed[0].track_id != original_id


def test_revival_max_age():
    # dead track removed after revivial_max_age, doesn't revive
    tracker = Tracker(min_hits=1, max_age=2, centroid_max_dist=50.0,
                       revival_max_age=3, appearance_threshold=0.5)
 
    frame = make_frame_with_box(RED, ORIGINAL_BOX)
    original_det = Detection(label="person", confidence=0.9, x1=20, y1=20, x2=80, y2=120)
    confirmed = tracker.update([original_det], frame=frame)
    original_id = confirmed[0].track_id

    # max_age + revivial_max_age = 3+3
    for _ in range(8):
        tracker.update([], frame=frame)

    reappear_frame = make_frame_with_box(RED, FAR_BOX)
    reappear_det = Detection(label="person", confidence=0.9, x1=200, y1=20, x2=260, y2=120)
    confirmed = tracker.update([reappear_det], frame=reappear_frame)

    assert confirmed[0].track_id != original_id