"""
A stress test to check how accuracy degrades under more realistic camera
conditions. Reuses the 200-image labeled set and boostrap-CI pipeline from
check_accuracy, applying corruption before detection.

Corruptions: brightness shift, Gaussian blur and JPEG re-compression.

Run using python -m scripts.check_robustness
"""
import cv2
import numpy as np

from scripts.check_accuracy import bootstrap_ci, evaluate, load_ground_truth


CONF_THRESHOLD = 0.4


def apply_brightness(frame: np.ndarray, factor: float) -> np.ndarray:
    """
    factor < 1 darkens, >1 brightens. Changes brightness suing linear
    scaling on raw BGR pixel values.
    """
    return np.clip(frame.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def apply_blur(frame: np.ndarray, kernel_size: int) -> np.ndarray:
    """Simulates blur (motion blur, out-of-focus, condensation etc). 
    kernel_size must be odd."""
    return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)


def apply_jpeg_compression(frame: np.ndarray, quality: int) -> np.ndarray:
    """
    Re-encodes and decodes through OpenCV's JPEG codec to simulate what
    network camera sends over wire. quality is 1-100. lower = more visible artifacts.
    """
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


# (corruption name, severity label, function)
# corruption_fn=None for clean baseline, then run through every corrupted variant
CORRUPTIONS = [
    ("clean", "none", None),
    ("dark", "mild", lambda f: apply_brightness(f, 0.6)),
    ("dark", "severe", lambda f: apply_brightness(f, 0.3)),
    ("bright", "mild", lambda f: apply_brightness(f, 1.4)),
    ("bright", "severe", lambda f: apply_brightness(f, 1.8)),
    ("blur", "mild", lambda f: apply_blur(f, 5)),
    ("blur", "severe", lambda f: apply_blur(f, 15)),
    ("jpeg", "mild", lambda f: apply_jpeg_compression(f, 30)),
    ("jpeg", "severe", lambda f: apply_jpeg_compression(f, 5)),
]


def main() -> None:
    ground_truths = load_ground_truth("coco_subset/labels.json")
    images_dir = "coco_subset/data"

    print(f"{'corruption':<10} {'severity':<8} {'precision [95% CI]':<28} {'recall [95% CI]':<28} n")
    for name, severity, corruption_fn in CORRUPTIONS:
        precision, recall, tp, fp, fn, per_image_results = evaluate(
            CONF_THRESHOLD, ground_truths, images_dir, corruption_fn=corruption_fn
        )
        p_point, p_lo, p_hi = bootstrap_ci(per_image_results, metric="precision")
        r_point, r_lo, r_hi = bootstrap_ci(per_image_results, metric="recall")

        p_str = f"{p_point:.3f} [{p_lo:.3f}, {p_hi:.3f}]"
        r_str = f"{r_point:.3f} [{r_lo:.3f}, {r_hi:.3f}]"
        print(f"{name:<10} {severity:<8} {p_str:<28} {r_str:<28} {len(per_image_results)}")


if __name__ == "__main__":
    main()