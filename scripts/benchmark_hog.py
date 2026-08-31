"""HOG detector latency benchmark, same methodology as benchmark.py (warmup,
then isolated per-call timing over many runs).

    python -m scripts.benchmark_hog

HOG's timing is dominated by how many sliding-window positions/scales get
scanned for a given frame size, not by pixel content, so a synthetic frame
is representative here (unlike the accuracy check, where real content
obviously matters).
"""

import time

import numpy as np

from app.detector import HogPersonDetector

NUM_WARMUP = 5
NUM_RUNS = 20


def main() -> None:
    detector = HogPersonDetector()
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    for _ in range(NUM_WARMUP):
        detector.detect(frame)

    latencies = []
    for _ in range(NUM_RUNS):
        t0 = time.perf_counter()
        detector.detect(frame)
        latencies.append(time.perf_counter() - t0)

    latencies = np.array(latencies) * 1000  # convert to ms
    mean_ms = float(np.mean(latencies))
    print(f"HOG: {mean_ms:.1f} ms mean, {float(np.percentile(latencies, 95)):.1f} ms p95, "
          f"{1000 / mean_ms:.1f} fps  ({NUM_RUNS} runs after {NUM_WARMUP} warmup, 640x480 frame)")


if __name__ == "__main__":
    main()