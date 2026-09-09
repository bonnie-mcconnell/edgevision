import numpy as np

from scripts.benchmark import RandomCalibrationDataReader


def test_yields_num_samples_then_none():
    reader = RandomCalibrationDataReader("input", (1,3, 4, 4), num_samples=3)

    samples = [reader.get_next(), reader.get_next(), reader.get_next()]
    assert all(s is not None for s in samples)
    assert reader.get_next() is None


def test_sample_shape_dtype_and_key_match_input():
    reader = RandomCalibrationDataReader("images", (1, 3, 640, 640), num_samples=1)

    sample = reader.get_next()

    assert sample is not None
    assert set(sample.keys()) == {"images"}
    assert sample["images"].shape == (1, 3, 640, 640)
    assert sample["images"].dtype == np.float32