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
import numpy as np

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


def evaluate(conf_threshold: float, ground_truths: dict, images_dir: str, verbose: bool = False) -> tuple[float, float, int, int, int, list[tuple[int, int, int]]]:
    detector = OnnxPersonDetector("models/yolov8n.onnx", 640, conf_threshold)
    total_tp, total_fp, total_fn = 0, 0, 0
    per_image_results: list[tuple[int, int, int]] = []  # (tp, fp, fn), one entry per image, the bootstrap resampling unit

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
        per_image_results.append((tp, fp, fn))

        if verbose:
            print(f"{filename}: {len(detections)} detected, {elapsed * 1000:.1f} ms")

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    return precision, recall, total_tp, total_fp, total_fn, per_image_results


def bootstrap_ci(
    per_image_results: list[tuple[int, int, int]],
    metric: str,
    num_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Bootstrap confidence interval for 'precision' or 'recall', resampling at
    the image level (each element of per_image_results is one image's
    (tp, fp, fn)).

    Returns (point_estimate, ci_low, ci_high), using the non-resampled
    aggregate as the point estimate and the resample distribution only for
    the interval.
    """
    assert metric in ("precision", "recall"), f"metric must be 'precision' or 'recall', got {metric!r}"
    n = len(per_image_results)
    assert n > 0, "need at least one image to bootstrap from"

    tp = np.array([r[0] for r in per_image_results], dtype=np.float64)
    fp = np.array([r[1] for r in per_image_results], dtype=np.float64)
    fn = np.array([r[2] for r in per_image_results], dtype=np.float64)

    # Point estimate using the real aggregate, not the mean of the resamples.
    # resamples are used to characterise spread around this number
    total_tp, total_fp, total_fn = tp.sum(), fp.sum(), fn.sum()
    if metric == "precision":
        point = float(total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
    else:
        point = float(total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0

    # Resample at image level by drawing n image-indices with replacement,
    # num_resamples times using (num_resamples, n) index matrix
    # Fancy-indexing tp/fp/fn by that matrix and summing along axis=1 gives 
    # every resample's totals in one vectorized go. 
    rng = np.random.default_rng(seed)
    resample_indices = rng.integers(0, n, size=(num_resamples, n))

    tp_r = tp[resample_indices].sum(axis=1)
    fp_r = fp[resample_indices].sum(axis=1)
    fn_r = fn[resample_indices].sum(axis=1)

    if metric == "precision":
        denom = tp_r + fp_r
    else:
        denom = tp_r + fn_r

    # A resample can draw only images with zero true positives, so guard against denominator=0
    numerator = tp_r
    metric_r = np.divide(numerator, denom, out=np.zeros_like(denom), where=denom > 0)

    alpha = 1 - confidence
    ci_low, ci_high = np.percentile(metric_r, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return point, float(ci_low), float(ci_high)


def main() -> None:
    ground_truths = load_ground_truth("coco_subset/labels.json")
    images_dir = "coco_subset/data"

    for threshold in [0.25, 0.4, 0.5, 0.6]:
        precision, recall, tp, fp, fn, per_image_results = evaluate(threshold, ground_truths, images_dir)
        p_point, p_lo, p_hi = bootstrap_ci(per_image_results, metric="precision")
        r_point, r_lo, r_hi = bootstrap_ci(per_image_results, metric="recall")
        print(
            f"conf={threshold}: P={p_point:.3f} [{p_lo:.3f}, {p_hi:.3f}]  "
            f"R={r_point:.3f} [{r_lo:.3f}, {r_hi:.3f}]  (tp={tp} fp={fp} fn={fn}, n={len(per_image_results)} images)"
        )


if __name__ == "__main__":
    main()