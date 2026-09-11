"""HOG and ONNX detectors behind one interface. Pick with DETECTOR_BACKEND."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    

class HogPersonDetector:
    """Classical HOG+SVM detector. No download required."""

    def __init__(self, target_width: int = 640):
        self.hog = cv2.HOGDescriptor()
        # opencv-python's .pyi stubs don't cover every auto-generated C++ binding
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()) # type: ignore[attr-defined]
        self.target_width = target_width

    def detect(self, frame: np.ndarray) -> tuple[list[Detection], float]:
        """Returns (detections, inference_time_seconds)."""
        h, w = frame.shape[:2]
        scale = self.target_width / w
        resized = cv2.resize(frame, (self.target_width, int(h * scale)))

        t0 = time.perf_counter()
        boxes, weights = self.hog.detectMultiScale(
            resized, winStride=(8, 8), padding=(8, 8), scale=1.05
        )
        elapsed = time.perf_counter() - t0

        detections = []
        for (x, y, bw, bh), weight in zip(boxes, weights):
            # weight is an SVM decision-function score, not a calibrated
            # probability, so squash it into a 0-1 range for callers that expect "confidence"
            confidence = float(min(1.0, max(0.0, weight / 2.0)))
            detections.append(Detection(
                label="person",
                confidence=confidence,
                x1=x / scale, y1=y / scale, # rescale all coordinate values
                x2=(x + bw) / scale, y2=(y + bh) / scale,
            ))
        return detections, elapsed


# standard 80-class COCO label order, index-matched to model's output
# no literal package/box/parcel class exists in COCO, closest available
# are backpack/suitcase/handbag. So this is approximation of package-monitoring
# feature, not production ready.
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

PACKAGE_CLASSES = {"backpack", "suitcase", "handbag"}


class OnnxDetector:
    """YOLOv8 onnx model via onnxruntime. Output shape is (1, 84, 8400) for COCO."""

    def __init__(self, model_path: str, 
                input_size: int = 640, 
                conf_threshold: float = 0.4,
                classes: set[str] | None = None,
        ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No ONNX model at {model_path}. Export one with:\n"
                f"  mkdir -p models && cd models\n"
                f"  yolo export model=yolov8n.pt format=onnx imgsz={input_size}\n"
                f"(yolo export writes the .onnx next to wherever yolov8n.pt was loaded from, "
                f"so cd into models/ first rather than moving the file after)"
            )
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size
        self.conf_threshold = conf_threshold

        # classes=None keeps existing callers (scripts) working the same/hardcoded
        # person only detection, same as before multi-class change
        classes = classes if classes is not None else {"person"}
        unknown = classes - set(COCO_CLASSES)
        if unknown:
            raise ValueError(f"Unknown COCO class name(s): {unknown}")
        self.class_ids = {COCO_CLASSES.index(name) for name in classes}

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        """
        Letterbox resize: scale to fit inside input_size x input_size, pad
        the rest with grey, rather than a straight resize that would
        distort the aspect ratio. Returns (tensor, scale, pad_x, pad_y) so
        detect() can map boxes back to original frame coordinates.
        """
        h, w = frame.shape[:2]
        scale = min(self.input_size / w, self.input_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))

        pad_x, pad_y = (self.input_size - new_w) // 2, (self.input_size - new_h) // 2
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[None, ...]  # HWC -> NCHW, add batch dim
        return img, scale, pad_x, pad_y

    def detect(self, frame: np.ndarray) -> tuple[list[Detection], float]:
        """Returns (detections, inference_time_seconds)."""
        inp, scale, pad_x, pad_y = self._preprocess(frame)

        t0 = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: inp})
        elapsed = time.perf_counter() - t0

        preds = outputs[0][0].T  # type: ignore[index]  # (8400, 84)

        # Vectorized, runs on every frame, bc 8400 rows in a Python for-loop 
        # would dwarf the model's own inference time on CPU.
        class_scores = preds[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_scores)), class_ids]
        keep_mask = np.isin(class_ids, list(self.class_ids)) & (confidences >= self.conf_threshold)

        boxes = preds[keep_mask, :4]
        confidences = confidences[keep_mask]
        kept_class_ids = class_ids[keep_mask]

        # undo letterbox padding, then the resize scale to get back to
        # original frame coordinates
        cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = (cx - bw / 2 - pad_x) / scale
        y1 = (cy - bh / 2 - pad_y) / scale
        x2 = (cx + bw / 2 - pad_x) / scale
        y2 = (cy + bh / 2 - pad_y) / scale

        detections = [
            Detection(COCO_CLASSES[cid], float(conf), float(a), float(b), float(c), float(d))
            for cid, conf, a, b, c, d in zip(kept_class_ids, confidences, x1, y1, x2, y2)
        ]
        return self._nms(detections), elapsed

    @staticmethod
    def _nms(detections: list[Detection], iou_threshold: float = 0.45) -> list[Detection]:
        """
        Non-max suppression: collapse overlapping boxes for the same object.
        Applied per label so that a person and a backpack can overlap in the same frame.
        Calls the single class version of this function safely.
        """
        if not detections:
            return []

        by_label: dict[str, list[Detection]] = {}
        for det in detections:
            by_label.setdefault(det.label, []).append(det)

        kept: list[Detection] = []
        for label_group in by_label.values():
            kept.extend(OnnxDetector._nms_single_class(label_group, iou_threshold))
        return kept

    @staticmethod
    def _nms_single_class(detections: list[Detection], iou_threshold: float = 0.45) -> list[Detection]:
        """Non-max suppression: collapse overlapping boxes for the same object."""  
        if not detections:
            return []
        
        boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in detections])
        scores = np.array([d.confidence for d in detections])
        order = scores.argsort()[::-1]

        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[1:][iou <= iou_threshold]
        
        return [detections[i] for i in keep]


def build_detector() -> HogPersonDetector | OnnxDetector:
    """Factory: picks the detector based on DETECTOR_BACKEND env var."""
    backend = os.environ.get("DETECTOR_BACKEND", "hog").strip().lower()
    if backend == "onnx":
        model_path = os.environ.get("ONNX_MODEL_PATH", "models/yolov8n.onnx")
        return OnnxDetector(model_path, classes={"person"} | PACKAGE_CLASSES)
    if backend != "hog":
        raise ValueError(f"Unknown DETECTOR_BACKEND: {backend!r}. Expected 'hog' or 'onnx'.")
    return HogPersonDetector()
