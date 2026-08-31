import pytest

from app.detector import Detection, OnnxPersonDetector


def make_det(x1, y1, x2, y2, conf):
    return Detection(label="person", confidence=conf, x1=x1, y1=y1, x2=x2, y2=y2)


def test_detection_is_frozen():
    det = make_det(0, 0, 10, 10, 0.9)
    with pytest.raises(Exception):
        det.confidence = 0.5


def test_nms_keeps_single_detection():
    dets = [make_det(0, 0, 10, 10, 0.9)]
    result = OnnxPersonDetector._nms(dets)
    assert len(result) == 1


def test_nms_suppresses_heavily_overlapping_box():
    # two boxes over almost the same region -> should collapse to 1
    dets = [
        make_det(0, 0, 100, 100, 0.9),
        make_det(2, 2, 98, 98, 0.7),
    ]
    result = OnnxPersonDetector._nms(dets)
    assert len(result) == 1
    assert result[0].confidence == 0.9  # higher-confidence box survives


def test_nms_keeps_non_overlapping_boxes():
    dets = [
        make_det(0, 0, 50, 50, 0.9),
        make_det(200, 200, 250, 250, 0.8),
    ]
    result = OnnxPersonDetector._nms(dets)
    assert len(result) == 2


def test_nms_empty_input():
    assert OnnxPersonDetector._nms([]) == []


def test_nms_partial_overlap_below_threshold_keeps_both():
    # boxes overlap only slightly -> IoU below default 0.45 threshold
    dets = [
        make_det(0, 0, 100, 100, 0.9),
        make_det(90, 90, 190, 190, 0.8),
    ]
    result = OnnxPersonDetector._nms(dets)
    assert len(result) == 2
