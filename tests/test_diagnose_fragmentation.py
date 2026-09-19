"""Tests for scripts/diagnose_fragmentation.py's death<->birth reassociation
matching.
Previously this matching was inline in run_fragmentation_analysis(). 
It's now match_deaths_to_births(), a pure function
of (death_events, birth_events) -> rows, so this file can pin
down its behavior without needing a real video or ONNX model.
"""
from scripts.diagnose_fragmentation import match_deaths_to_births


def test_nearby_birth_after_death_is_flagged_reassociated():
    death_events = [(1, 100, (200, 200, 260, 400))]
    birth_events = [(2, 110, (205, 205, 265, 405))]  # 10 frames later, ~7px away

    rows = match_deaths_to_births(death_events, birth_events, max_frames=45, max_dist=150.0)

    assert len(rows) == 1
    assert rows[0]["reassociated"] is True
    assert rows[0]["born_id"] == 2
    assert rows[0]["gap_frames"] == 10


def test_birth_outside_time_window_is_not_flagged():
    death_events = [(1, 100, (200, 200, 260, 400))]
    birth_events = [(2, 200, (205, 205, 265, 405))]  # 100 frames later, outside the window

    rows = match_deaths_to_births(death_events, birth_events, max_frames=45, max_dist=150.0)

    assert rows[0]["reassociated"] is False
    assert rows[0]["born_id"] is None


def test_birth_too_far_away_is_not_flagged():
    death_events = [(1, 100, (200, 200, 260, 400))]
    birth_events = [(2, 110, (900, 900, 960, 1100))]  # right after, but nowhere near

    rows = match_deaths_to_births(death_events, birth_events, max_frames=45, max_dist=150.0)

    assert rows[0]["reassociated"] is False


def test_one_birth_cannot_be_claimed_by_two_deaths():
    """Two deaths near the same birth event used to both get
    flagged as reassociated with it, double-counting one real re-appearance
    as two separate 'plausible fragmentations'."""
    death_events = [
        (1, 100, (200, 200, 260, 400)),
        (2, 102, (210, 200, 270, 400)),  # a different track, dies 2 frames later, very close by
    ]
    birth_events = [
        (3, 110, (205, 205, 265, 405)),  # only one birth, roughly between the two deaths
    ]

    rows = match_deaths_to_births(death_events, birth_events, max_frames=45, max_dist=150.0)

    reassociated = [r for r in rows if r["reassociated"]]
    assert len(reassociated) == 1 
    # the closer death (smaller centroid distance) should be the one that wins the match
    assert reassociated[0]["died_id"] == 1


def test_closest_death_wins_no_order():
    """Global closest-pairs-first assignment, not first-death-processed-wins:
    death 2 arrives later in the list but is the closer match to the only birth."""
    death_events = [
        (1, 100, (0, 0, 60, 200)),      # far from the birth below
        (2, 100, (200, 200, 260, 400)),  # right next to the birth below
    ]
    birth_events = [
        (3, 110, (205, 205, 265, 405)),
    ]

    rows = match_deaths_to_births(death_events, birth_events, max_frames=45, max_dist=1000.0)

    reassociated = [r for r in rows if r["reassociated"]]
    assert len(reassociated) == 1
    assert reassociated[0]["died_id"] == 2


def test_no_deaths_or_births_returns_empty():
    assert match_deaths_to_births([], []) == []


def test_multiple_independent_pairs_all_match():
    death_events = [
        (1, 100, (0, 0, 60, 200)),
        (2, 300, (500, 500, 560, 700)),
    ]
    birth_events = [
        (3, 110, (5, 5, 65, 205)),
        (4, 310, (505, 505, 565, 705)),
    ]

    rows = match_deaths_to_births(death_events, birth_events, max_frames=45, max_dist=150.0)

    assert sum(1 for r in rows if r["reassociated"]) == 2