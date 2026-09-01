# EdgeVision

Person detection + zone alerting for a security camera use case, with a benchmark comparing FP32 vs INT8 quantized inference on CPU.

**Live:** [edgevision-mpll.onrender.com/health](https://edgevision-mpll.onrender.com/health) · [interactive API docs](https://edgevision-mpll.onrender.com/docs)
(runs the HOG backend on Render's free tier, no camera attached, so `/ws/detections` has nothing to stream, see "Running it" below for testing detection with a real feed)

![demo](results/demo.gif)
*Live ONNX detection on a real street-scene clip with bounding boxes, per-detection confidence, and per-frame inference time. Full video: `results/video/street_annotated.mp4`, generated via `python -m scripts.test_video`.*

Video frame arrives, a detector finds people, boxes get checked against a zone, and alerts get deduped through Redis before hitting a websocket. There's also a separate benchmark script for testing whether quantizing a model helps on CPU.

## Accuracy

Evaluated `OnnxPersonDetector` (real yolov8n.onnx) against a 40-image labeled subset of COCO val2017 (person class only, seeded sample. See `scripts/fetch_coco_subset.py`), matching predictions to ground truth via IoU ≥ 0.5:

| conf threshold | precision | recall | tp | fp | fn |
|---|---|---|---|---|---|
| 0.25 | 0.797 | 0.615 | 110 | 28 | 69 |
| 0.40 (default) | 0.877 | 0.559 | 100 | 14 | 79 |
| 0.50 | 0.917 | 0.492 | 88 | 8 | 91 |
| 0.60 | 0.974 | 0.413 | 74 | 2 | 105 |

Clean, monotonic precision/recall tradeoff as the threshold rises: fewer false alarms, but more missed people. For a security-camera use case specifically, a missed person is usually worse than a false alarm, which argues for running below the library default of 0.4, closer to 0.25, trading some false-positive noise for meaningfully better recall.

Caveat: 40 images is a small sample. The curve shape is consistent and direction trustworthy but treat the exact precision/recall values as approximate. `coco_subset/` itself is gitignored, to regenerate it use `python scripts/fetch_coco_subset.py`.

Two annotated real-image spot checks from a previous check_accuracy are also included as a quick visual sanity check (`results/examples/`): `bus.jpg` (4 people in frame, all 4 found) and `zidane.jpg` (2 people, both found).

## Benchmark

| variant | size (KB) | mean latency (ms) | p95 (ms) | fps |
|---|---|---|---|---|
| fp32 | 114.3 | 1.38 | 1.736 | 722.2 |
| int8 | 33.6 | 3.83 | 4.65 | 261.0 |

Size dropped 70% like you'd expect. Latency didn't improve though, it got worse. Dynamic quantization adds a dequantize step around every op, and on a small model that overhead is bigger than what you save doing the matmuls in int8. So "quantize it, it'll be faster" isn't universally true, it depends on the model being deep enough that the compute savings outweigh the added ops. Static quantization with a calibration set would probably behave differently, but I haven't tested that yet.

This benchmark runs on a tiny CNN built by hand with `onnx.helper` instead of a real yolov8n, mainly because pulling in torch + a full pretrained model felt like overkill for testing whether the quantize/benchmark code works. The methodology's the same: swap `MODEL_PATH` once you've exported a real one and bump `input_shape` to 640x640. Weights are seeded so the model itself is identical every run, but the absolute latency numbers move around depending on what else is running on the machine at the time, with swings of 2x+ between a quiet machine and a loaded one. The size reduction and the direction of the latency result (int8 slower, not faster) held steady across every run though.

HOG on its own runs about 154ms/frame (roughly 6.5fps) at 640px on a regular CPU, averaged over 20 runs after warmup, for reference.

### Same benchmark, on the real yolov8n

Once yolov8n.onnx was actually exported (see below) and the same FP32 vs INT8 comparison reran against it instead of the hand-built demo model, the result flipped:

| variant | size (KB) | mean latency (ms) | fps |
|---|---|---|---|
| fp32 | 12550 | 94.3 | 10.6 |
| int8 | 3422 | 65.6 | 15.2 |

INT8 is 1.44x faster here, because the dequant overhead is roughly fixed per op, but the compute it's saving scales with the model's actual size. On a network this small (a few thousand params) that overhead dominates; on yolov8n (3.1M params, 8.7 GFLOPs) there's enough real matmul work that quantizing it is a net win. This supports the idea that the model's compute density is what determines whether quantization helps or hurts latency. See `results/benchmark_real_model.csv`.

The same comparison on different CPU hardware (AMD Ryzen 5 7520U vs. the machine above) came out the other way, int8 slightly *slower* (356ms vs 403ms fp32, 0.88x). Quantization's payoff depends on the specific CPU's int8 vs fp32 throughput characteristics, not just the model. 

## Architecture / design decisions

- **Two detectors behind one interface:** (`app/detector.py`): `HogPersonDetector` (OpenCV's built-in HOG+SVM, zero setup, runs by default) and `OnnxPersonDetector` (real YOLOv8 via onnxruntime), both returning the same `Detection` dataclass and `(detections, elapsed_time)` tuple, swappable via `DETECTOR_BACKEND` env var.
- **Letterbox resize, not stretch-resize:**, before feeding frames to the ONNX model the detector service scales to fit while preserving aspect ratio, pads the rest with grey, then undoes the scale+pad math on the output boxes. A straight resize would distort people's proportions and hurt accuracy.
- **NMS threshold of 0.45:** for collapsing duplicate overlapping boxes for the same person, applied after class-filtering to person-only, vectorized with NumPy rather than a per-box Python loop (8400 candidate boxes per frame would otherwise far exceed the model's own inference time on CPU).
- **Redis for alert dedup**, via `SET NX EX` (atomic "set this key only if it doesn't already exist, with a TTL") giving a 30-second cooldown per zone+label with no separate cleanup job needed. The key just expires on its own.
- **Dependency injection via FastAPI's `Depends()`:** (`app/dependencies.py`): the detector, Redis client, and alert manager are constructed through cached factory functions rather than bare module-level globals, so tests can cleanly swap in fakes (`fakeredis`, a stub detector) via `app.dependency_overrides` without touching route code. A `lifespan` context manager eagerly builds and pings these at startup (fail fast if the model file's missing or Redis is unreachable) rather than lazily on the first request.

## Tests

23 tests, `pip install pytest fakeredis && pytest tests/ -v`:
- `test_detector.py` tests NMS (suppression, survival, empty input, partial overlap below threshold)
- `test_alerts.py` tests `Zone.overlaps_box`'s four separation directions plus edge-touching, `AlertManager`'s cooldown/dedup logic via `fakeredis`
- `test_main.py` tests FastAPI route coverage (`/health`, `/alerts/recent`) via `TestClient`, with a pytest fixture that patches both the `lifespan` startup calls and the `Depends()`-resolved dependencies so tests never touch a real model file or Redis instance

## Running it

```
docker compose up --build
curl http://localhost:8000/health
```

For a real webcam it's easier to skip Docker (webcam passthrough is a pain cross-platform) and just run it locally. Needs Redis installed (`brew install redis` on Mac, `apt install redis-server` on Ubuntu):

```
pip install -r requirements.txt
redis-server &
VIDEO_SOURCE=0 uvicorn app.main:app --reload
```

then connect to `ws://localhost:8000/ws/detections` with a websocket client (a browser address bar won't work because it can't open a websocket connection via plain navigation, so you'll get a 404 there. Instead use something like `websocat` or a small Python script instead).

For the ONNX backend:
```
pip install ultralytics
cd models && yolo export model=yolov8n.pt format=onnx imgsz=640
```
The Dockerfile only bundles `app/`, not `models/`, since the default HOG backend doesn't need a model file. If you want the ONNX backend in Docker you'll need to mount `models/` as a volume rather than rely on the build.

### Deployed version

The live Render deployment (linked at the top) runs the HOG backend with a managed Redis instance, provisioned via `render.yaml` as a Blueprint. No `VIDEO_SOURCE` is set (no camera on a cloud server), so `/health`, `/docs`, `/alerts/recent`, and `/benchmark/results` all work; `/ws/detections` has nothing to stream.

## What this isn't

This is a benchmarking harness and an alerting service, not an on-device deployment. It runs on a regular CPU through onnxruntime and doesn't target any specific camera SoC, NPU, or cross-compiled runtime.

## TODO / known gaps

- HOG is not very accurate, it's there because it needs no download. For anything real, I would swap to the ONNX path.
- accuracy eval is on a 40-image sample, which is worth widening for tighter confidence intervals
- no auth on the websocket
- alert history is just whatever's in Redis's bounded list, nothing persisted long term
- single camera only right now
- want to try static quantization + FP16 and see if the int8-helps result holds up further
- zone is hardcoded (`DEFAULT_ZONE` in `main.py`). A real deployment would need this per-camera and user-configurable