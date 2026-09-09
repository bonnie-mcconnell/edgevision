import random
import pytest

from scripts.check_accuracy import bootstrap_ci


def naive_bootstrap_ci(per_image_results, metric, num_resamples, confidence=0.95, seed=7):
    """
    Independent non-vectorized reference implementation (different RNG) 
    used only to cross-check bootstrap_ci's vectorized numpy version is correct.
    """
    random.seed(seed)
    n = len(per_image_results)
    total_tp = sum(r[0] for r in per_image_results)
    total_fp = sum(r[1] for r in per_image_results)
    total_fn = sum(r[2] for r in per_image_results)
    point = total_tp / (total_tp + total_fp) if metric == "precision" else total_tp / (total_tp + total_fn)

    vals = []
    for _ in range(num_resamples):
        sample = random.choices(per_image_results, k=n)
        tp = sum(r[0] for r in sample)
        fp = sum(r[1] for r in sample)
        fn = sum(r[2] for r in sample)
        denom = (tp + fp) if metric == "precision" else (tp + fn)
        vals.append(tp / denom if denom > 0 else 0.0)
    vals.sort()
    alpha = 1 - confidence
    lo = vals[int(len(vals) * alpha / 2)]
    hi = vals[int(len(vals) * (1 - alpha / 2))]
    return point, lo, hi


def test_ci_collapses_to_point_when_all_images_identical():
    """If every image has the same (tp, fp, fn), every possible resample
    sums to exactly n times that constant so the CI must have zero width."""
    identical = [(5, 1, 1)] * 10
    point, lo, hi = bootstrap_ci(identical, metric="precision")

    assert point == lo == hi
    assert point == 5 / 6


def test_ci_collapses_to_point_with_single_image():
    """With n=1, every resample draws the same image, the CI must be zero-width."""
    single = [(3, 1, 2)]
    point, lo, hi = bootstrap_ci(single, metric="recall")

    assert point == lo == hi
    assert point == 3 / 5


def test_point_estimate_is_real_aggregate_not_resample_mean():
    """Point estimate should be exactly tp/(tp+fp) (or tp/(tp+fn)) computed
    on data independent of num_resamples or seed."""
    per_image = [(3, 0, 1), (2, 1, 0), (0, 0, 2), (4, 1, 1), (1, 0, 0)]
    total_tp, total_fp, total_fn = 10, 2, 4

    point_precision, _, _ = bootstrap_ci(per_image, metric="precision", num_resamples=50, seed=1)
    point_recall, _, _ = bootstrap_ci(per_image, metric="recall", num_resamples=999, seed=2)

    assert point_precision == total_tp / (total_tp + total_fp)
    assert point_recall == total_tp / (total_tp + total_fn)


def test_bootstrap_ci_matches_independent_naive_implementation():
    """Cross-check against a separate, non-vectorized
    implementation using a different RNG. Agreement here shows the
    vectorized numpy math is correct and not just internally
    consistent with itself."""
    per_image = [(3, 0, 1), (2, 1, 0), (0, 0, 2), (4, 1, 1), (1, 0, 0)]

    for metric in ("precision", "recall"):
        vec_point, vec_lo, vec_hi = bootstrap_ci(per_image, metric=metric, num_resamples=20000, seed=42)
        naive_point, naive_lo, naive_hi = naive_bootstrap_ci(per_image, metric=metric, num_resamples=20000)

        assert vec_point == naive_point  # both are the same deterministic aggregate
        assert abs(vec_lo - naive_lo) < 0.02
        assert abs(vec_hi - naive_hi) < 0.02


def test_wider_sample_produces_tighter_interval():
    """More images (same underlying per-image tp/fp/fn distribution,
    just repeated) should shrink the CI width."""
    small = [(3, 0, 1), (2, 1, 0), (0, 0, 2), (4, 1, 1), (1, 0, 0)]
    large = small * 20

    _, lo_small, hi_small = bootstrap_ci(small, metric="precision", num_resamples=5000)
    _, lo_large, hi_large = bootstrap_ci(large, metric="precision", num_resamples=5000)

    assert (hi_large - lo_large) < (hi_small - lo_small)


def test_zero_denominator_resample_handled_without_crashing():
    """If a resample draws only zero-tp images, precision's
    denominator can be zero for that resample. Should resolve to 0.0 for
    that draw and not raise."""
    all_missed = [(0, 0, 3), (0, 0, 5)]  # never any detections at all

    point, lo, hi = bootstrap_ci(all_missed, metric="precision", num_resamples=500)

    assert point == 0.0
    assert lo == 0.0
    assert hi == 0.0


def test_same_seed_is_reproducible():
    per_image = [(3, 0, 1), (2, 1, 0), (0, 0, 2), (4, 1, 1), (1, 0, 0)]

    result_a = bootstrap_ci(per_image, metric="recall", num_resamples=500, seed=42)
    result_b = bootstrap_ci(per_image, metric="recall", num_resamples=500, seed=42)

    assert result_a == result_b


def test_invalid_metric_raises():
    with pytest.raises(AssertionError):
        bootstrap_ci([(1, 0, 0)], metric="f1")