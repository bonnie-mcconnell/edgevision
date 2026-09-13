import numpy as np

from scripts.check_robustness import apply_blur, apply_brightness, apply_jpeg_compression, CORRUPTIONS


def test_apply_brightness_darkens():
    frame = np.full((10, 10, 3), 200, dtype=np.uint8)
    darkened = apply_brightness(frame, 0.5)
    assert darkened[0, 0, 0] == 100


def test_apply_brightness_and_clips():
    frame = np.full((10, 10, 3), 200, dtype=np.uint8)
    brightened = apply_brightness(frame, 2.0)
    assert brightened[0, 0, 0] == 255 # 400 would overflow so clips


def test_apply_blur():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    frame[25:, :] = 255 # horizontal edge at row 25
    blurred = apply_blur(frame, 15)
    edge_value = blurred[25, 25, 0]
    assert 0 < edge_value < 255 # edge is no longer sharp


def test_apply_jpeg_compression():
    frame = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    compressed = apply_jpeg_compression(frame, quality=5)
    assert compressed.shape == frame.shape
    assert not np.array_equal(compressed, frame)


def test_corruptions_list_contains_clean_baseline():
    names = [c[0] for c in CORRUPTIONS]
    assert "clean" in names
    clean_entry = next(c for c in CORRUPTIONS if c[0] == "clean")
    assert clean_entry[2] is None


def test_all_corruption_entries_correct_form():
    for name, severity, fn in CORRUPTIONS:
        assert isinstance(name, str) and name
        assert isinstance(severity, str) and severity
        assert fn is None or callable(fn)