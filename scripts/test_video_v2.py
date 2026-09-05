import os
import time
import cv2

from app.detector import OnnxPersonDetector, draw_detections
from app.tracker import Tracker, draw_tracks

VIDEO_SOURCE = "test_footage/street.mp4"
OUTPUT_DIR = "results/video"


def open_writer(out_path: str, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    for codec in ("avc1", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*codec) # type:ignore[attr-defined]
        writer = cv2.VideoWriter(out_path, fourcc, fps, size)
        if writer.isOpened():
            print(f"Using codec: {codec}")
            return writer
    raise SystemExit(f"Couldn't open writer for {out_path} with any known codec")


def main() -> None:
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise SystemExit(f"Couldn't open {VIDEO_SOURCE}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    detector = OnnxPersonDetector("models/yolov8n.onnx")
    tracker = Tracker()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    raw_path = os.path.join(OUTPUT_DIR, "street_annotated.mp4")
    tracked_path = os.path.join(OUTPUT_DIR, "street_tracked.mp4")
    raw_writer = open_writer(raw_path, fps, (width, height))
    tracked_writer = open_writer(tracked_path, fps, (width, height))

    frame_idx = 0
    start = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            detections, elapsed = detector.detect(frame)
            tracks = tracker.update(detections)

            # two seperate copies, both draw_detections/draw_tracks mutate 
            # in place so need before/after frames to stay independent
            raw_frame = draw_detections(frame.copy(), detections)
            cv2.putText(raw_frame, f"{len(detections)} detected (no tracking)  |  {elapsed * 1000:.0f}ms",
                        (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            raw_writer.write(raw_frame)

            tracked_frame = draw_tracks(frame.copy(), tracks)
            cv2.putText(tracked_frame, f"{len(tracks)} tracked  |  {elapsed * 1000:.0f}ms",
                        (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            tracked_writer.write(tracked_frame)

            if frame_idx % 60 == 0:
                print(f"frame {frame_idx}/{total_frames}")

    finally:
        cap.release()
        raw_writer.release()
        tracked_writer.release()

    total_elapsed = time.perf_counter() - start
    print(f"Done: {frame_idx} frames in {total_elapsed:.1f}s -> {raw_path}, {tracked_path}")


if __name__ == "__main__":
    main()