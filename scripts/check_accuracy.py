"""Evaluates OnnxPersonDetector's precision/recall against a labeled COCO
val2017 person subset, sweeping confidence thresholds to show the
precision/recall tradeoff.

Requires coco_subset/ to exist (run scripts/fetch_coco_subset.py first)
and models/yolov8n.onnx to exist (see README for the export command).

    python -m scripts.check_accuracy
"""

import os
import json

import cv2

from app.detector import OnnxPersonDetector, Detection


def iou(box_a: tuple, box_b: tuple) -> float:
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection_area = (ix2 - ix1) * (iy2 - iy1)
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    return intersection_area / (box_a_area + box_b_area - intersection_area)


def load_ground_truth(labels_path: str) -> dict[str, list[tuple]]:
    with open(labels_path) as f:
        data = json.load(f)

    person_category_id = None
    for cat in data["categories"]:
        if cat["name"] == "person":
            person_category_id = cat["id"]
    assert person_category_id is not None, "no 'person' category found in labels.json"
    
    id_to_filename = {}
    for img in data["images"]:
        id_to_filename[img["id"]] = img["file_name"]

    result = {}
    for ann in data["annotations"]:
        if ann["category_id"] == person_category_id:
            bbox = ann["bbox"]
            x2 = bbox[0] + bbox[2]
            y2 = bbox[1] + bbox[3]
            box = (bbox[0], bbox[1], x2, y2)
            filename = id_to_filename[ann["image_id"]]
            result.setdefault(filename, []).append(box)

    return result


def match_detections(detections: list[Detection], ground_truth_boxes: list[tuple], iou_threshold: float = 0.5) -> tuple[int, int, int]:
    used_gt_indices = set()
    tp = 0
    fp = 0

    for d in detections:
        d_box = (d.x1, d.y1, d.x2, d.y2)
        best_iou = 0.0
        best_gt_idx = None

        for idx, g_box in enumerate(ground_truth_boxes):
            if idx in used_gt_indices:
                continue  # already claimed by a higher-confidence prediction
            box_iou = iou(d_box, g_box)

            if box_iou > best_iou:
                best_iou = box_iou
                best_gt_idx = idx

        if best_gt_idx is not None and best_iou >= iou_threshold:
            used_gt_indices.add(best_gt_idx)
            tp += 1
        else:
            fp += 1

    fn = len(ground_truth_boxes) - len(used_gt_indices)
    return tp, fp, fn


def evaluate(conf_threshold: float, ground_truths: dict, images_dir: str, verbose: bool = False) -> tuple[float, float, int, int, int]:
    detector = OnnxPersonDetector("models/yolov8n.onnx", 640, conf_threshold)
    total_tp, total_fp, total_fn = 0, 0, 0

    for filename in os.listdir(images_dir):
        src_path = os.path.join(images_dir, filename)
        frame = cv2.imread(src_path)
        if frame is None:
            print(f"Skipping {filename}: couldn't read {src_path}")
            continue

        detections, elapsed = detector.detect(frame)

        gt_boxes = ground_truths.get(filename, [])
        tp, fp, fn = match_detections(detections, gt_boxes)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        if verbose:
            print(f"{filename}: {len(detections)} detected, {elapsed * 1000:.1f} ms")

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    return precision, recall, total_tp, total_fp, total_fn


def main() -> None:
    ground_truths = load_ground_truth("coco_subset/labels.json")
    images_dir = "coco_subset/data"
    
    for threshold in [0.25, 0.4, 0.5, 0.6]:
        precision, recall, tp, fp, fn = evaluate(threshold, ground_truths, images_dir)
        print(f"conf={threshold}: P={precision:.3f} R={recall:.3f} (tp={tp} fp={fp} fn={fn})")


if __name__ == "__main__":
    main()