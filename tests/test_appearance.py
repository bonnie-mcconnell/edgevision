import numpy as np

from app.appearance import appearance_descriptor, appearance_distance


def make_color_frame(size=200):
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[20:120, 20:80] = (30, 30, 200) # red region
    frame[20:120, 100:160] = (200, 30, 30) # blue region
    return frame


RED_BOX = (20, 20, 80, 120)
BLUE_BOX = (100, 20, 160, 120)


def test_appearance_histogram_sums_to_one():
    frame = make_color_frame()
    hist = appearance_descriptor(frame, RED_BOX)
    assert hist is not None
    assert abs(hist.sum() - 1.0) < 1e-6


def test_appearance_descriptor_matches_itself():
    frame = make_color_frame()
    hist = appearance_descriptor(frame, RED_BOX)
    assert appearance_distance(hist, hist) == 0


def test_appearance_descriptor_color_distance():
    frame = make_color_frame()
    h_red = appearance_descriptor(frame, RED_BOX)
    h_blue = appearance_descriptor(frame, BLUE_BOX)
    assert appearance_distance(h_red, h_blue) == 1.0


def test_appearance_descriptor_noisy_same_color():
    # noisy version of same appearance must be closer than a different
    # appearance is
    frame = make_color_frame()
    noisy_frame = frame.copy()
    rng = np.random.default_rng(0)
    region = noisy_frame[20:120, 20:80].astype(np.int16)
    region = np.clip(region + rng.integers(-15, 15, region.shape), 0, 255)
    noisy_frame[20:120, 20:80] = region.astype(np.uint8)

    h_red = appearance_descriptor(frame, RED_BOX)
    h_red_noisy = appearance_descriptor(noisy_frame, RED_BOX)
    h_blue = appearance_descriptor(frame, BLUE_BOX)

    same_person_dist = appearance_distance(h_red, h_red_noisy)
    diff_person_dist = appearance_distance(h_red, h_blue)

    assert same_person_dist < diff_person_dist


def test_appearance_descriptor_out_of_frame_box():
    frame = make_color_frame()
    assert appearance_descriptor(frame, (500, 500, 600, 600)) is None


def test_appearance_descriptor_none_for_zero_area_box():
    frame = make_color_frame()
    assert appearance_descriptor(frame, (50, 50, 50, 120)) is None


def test_appearance_descriptor_clips():
    # box straddling frame edge should still describe whats inside frame
    frame = make_color_frame()
    partially_outside = (-50, 20, 80, 120) # left edge out
    hist = appearance_descriptor(frame, partially_outside)
    assert hist is not None
    assert abs(hist.sum() - 1.0) < 1e-6


def test_appearance_distance_descriptor_missing():
    frame = make_color_frame()
    hist = appearance_descriptor(frame, RED_BOX)
    assert appearance_distance(hist, None) == 1.0
    assert appearance_distance(None, hist) == 1.0
    assert appearance_distance(None, None) == 1.0
