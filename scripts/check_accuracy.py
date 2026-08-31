"""Runs OnnxPersonDetector against two standard CV test images bundled with
the ultralytics package (no network fetch needed) and saves annotated output
as visual proof of real & working detection.

    python -m scripts.check_accuracy

Requires models/yolov8n.onnx to exist (see README for the export command)
and ultralytics installed (only for its bundled sample images - the actual
detector never imports ultralytics, only onnxruntime).
"""

import os

import cv2

from app.detector import OnnxPersonDetector

TEST_IMAGES = ["bus.jpg", "zidane.jpg"]


def find_ultralytics_assets() -> str:
    import ultralytics
    return os.path.join(os.path.dirname(ultralytics.__file__), "assets")


def main() -> None:
    assets_dir = find_ultralytics_assets()
    detector = OnnxPersonDetector("models/yolov8n.onnx")
    os.makedirs("results/examples", exist_ok=True)

    for name in TEST_IMAGES:
        src_path = os.path.join(assets_dir, name)
        frame = cv2.imread(src_path)
        if frame is None:
            print(f"skipping {name}, couldn't read {src_path}")
            continue

        detections, elapsed = detector.detect(frame)
        print(f"{name}: {len(detections)} person(s) found, {elapsed * 1000:.1f}ms")

        for d in detections:
            x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, f"person {d.confidence:.2f}", (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            print(f"  conf={d.confidence:.2f} box=({x1},{y1},{x2},{y2})")

        out_path = f"results/examples/{name.replace('.jpg', '_annotated.jpg')}"
        cv2.imwrite(out_path, frame)
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()