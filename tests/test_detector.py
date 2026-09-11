import pytest

from app.detector import Detection, OnnxDetector, COCO_CLASSES


DEMO_MODEL_PATH = "models/demo_fp32.onnx"  # any valid onnx file works for constructor-level tests
# __init__ only needs a loadable model to build a session, it doesn't need
# 84-channel COCO output shape unless detect() is actually called.

def make_det(x1, y1, x2, y2, conf, label="person"):
    return Detection(label, confidence=conf, x1=x1, y1=y1, x2=x2, y2=y2)


def test_detection_is_frozen():
    det = make_det(0, 0, 10, 10, 0.9)
    with pytest.raises(Exception):
        det.confidence = 0.5


def test_nms_keeps_single_detection():
    dets = [make_det(0, 0, 10, 10, 0.9)]
    result = OnnxDetector._nms(dets)
    assert len(result) == 1


def test_nms_suppresses_heavily_overlapping_box():
    # two boxes over almost the same region -> should collapse to 1
    dets = [
        make_det(0, 0, 100, 100, 0.9),
        make_det(2, 2, 98, 98, 0.7),
    ]
    result = OnnxDetector._nms(dets)
    assert len(result) == 1
    assert result[0].confidence == 0.9  # higher-confidence box survives


def test_nms_keeps_non_overlapping_boxes():
    dets = [
        make_det(0, 0, 50, 50, 0.9),
        make_det(200, 200, 250, 250, 0.8),
    ]
    result = OnnxDetector._nms(dets)
    assert len(result) == 2


def test_nms_empty_input():
    assert OnnxDetector._nms([]) == []


def test_nms_partial_overlap_below_threshold_keeps_both():
    # boxes overlap only slightly -> IoU below default 0.45 threshold
    dets = [
        make_det(0, 0, 100, 100, 0.9),
        make_det(90, 90, 190, 190, 0.8),
    ]
    result = OnnxDetector._nms(dets)
    assert len(result) == 2


def test_nms_doesnt_suppress_overlapping_different_classes():
    dets = [
        make_det(0, 0, 100, 100, 0.9, label="person"),
        make_det(2, 2, 98, 98, 0.85, label="backpack"),
    ]
    result = OnnxDetector._nms(dets)
    assert len(result) == 2
    assert {d.label for d in result} == {"person", "backpack"}


def test_nms_suppresses_within_class_when_mixed():
    # two overlapping persons + two overlapping backpacks should collapse independently
    dets = [
        make_det(0, 0, 100, 100, 0.9, label="person"),
        make_det(2, 2, 98, 98, 0.7, label="person"),
        make_det(300, 300, 400, 400, 0.8, label="backpack"),
        make_det(302, 302, 398, 398, 0.6, label="backpack"),
    ]
    result = OnnxDetector._nms(dets)
    assert len(result) == 2
    labels = [d.label for d in result]
    assert labels.count("person") == 1
    assert labels.count("backpack") == 1


def test_default_classes_person_only():
    detector = OnnxDetector(DEMO_MODEL_PATH, classes=None)
    assert detector.class_ids == {COCO_CLASSES.index("person")}


def test_custom_classes_resolve_to_correct_coco_ids():
    detector = OnnxDetector(DEMO_MODEL_PATH, classes={"backpack", "suitcase"})
    assert detector.class_ids == {COCO_CLASSES.index("backpack"), COCO_CLASSES.index("suitcase")}


def test_unknown_class_name_raises():
    with pytest.raises(ValueError):
        OnnxDetector(DEMO_MODEL_PATH, classes={"unknown_class"})