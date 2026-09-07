# EdgeVision

Person detection + zone alerting for a security camera use case, with a benchmark comparing FP32 vs INT8 quantized inference on CPU.

## Quick start

1. [/demo](https://edgevision-mpll.onrender.com/demo): upload any photo with people in it, see the real ONNX YOLOv8 model detect them live with no setup needed.
2. [/docs](https://edgevision-mpll.onrender.com/docs): interactive API reference, try any endpoint directly in the browser.
3. [/live](https://edgevision-mpll.onrender.com/live): real-time webcam view (only works when running the server locally with a webcam attached, the deployed version has no camera, so this will show a connection error there).

The deployed instance is on Render's free tier, which spins down after inactivity, so the first request after a while may take 20-30s to wake up.

## What it does

A video frame arrives, a detector finds people, a lightweight tracker links detections across frames into persistent identities, tracked boxes get checked against a zone, and alerts get deduped through Redis before hitting a websocket. This includes a standalone image-upload endpoint for zero-setup testing, a live in-browser view for local webcam testing, and a separate benchmark script for testing whether quantizing a model helps on CPU.

## Accuracy

Evaluated `OnnxPersonDetector` (real yolov8n.onnx) against a 40-image labeled subset of COCO val2017 (person class only, seeded sample, see `scripts/fetch_coco_subset.py`), matching predictions to ground truth via IoU ≥ 0.5:

| conf threshold | precision | recall | tp | fp | fn |
|---|---|---|---|---|---|
| 0.25 | 0.797 | 0.615 | 110 | 28 | 69 |
| 0.40 (default) | 0.877 | 0.559 | 100 | 14 | 79 |
| 0.50 | 0.917 | 0.492 | 88 | 8 | 91 |
| 0.60 | 0.974 | 0.413 | 74 | 2 | 105 |

There is a monotonic precision/recall tradeoff as the threshold rises with fewer false alarms, but more missed people. For a security-camera use case a missed person is usually worse than a false alarm, which argues for running below the library default of 0.4, closer to 0.25, trading some false-positive noise for meaningfully better recall.

However, 40 images is a small sample, so although the curve shape is consistent and direction looks correct treat the exact precision/recall values as approximate. `coco_subset/` is gitignored, can regenerate it with `python scripts/fetch_coco_subset.py`.

Two annotated real-image spot checks are included: (`results/examples/`): `bus.jpg` (4 people in frame, all 4 found) and `zidane.jpg` (2 people, both found).

### Observations

- **A false positive on a tall, narrow, high-contrast object:** a red/white striped traffic bollard got boxed as `person`:

  ![false positive on a sign](results/examples/sign_false_positive.png)

  Consistent with the aggregate false-positive rate from the accuracy table above. A real deployment would want either a higher confidence threshold in crowded scenes, or a second-stage filter (like minimum aspect ratio) to catch shape outliers like this.

- **Zero detections on extremely dense crowd photos:** Uploading a wide festival-crowd photo (hundreds of people, each only a few pixels tall after the model's 640x640 letterbox resize) to `/demo/detect` returns no boxes at all. This is because single-shot detectors like YOLO lose the ability to detect objects below a certain pixel footprint, and is why dedicated crowd-counting models exist (density estimation rather than per-instance boxes). To improve crowded image detection, use tiled/sliding-window inference, in which you split the image into overlapping crops, detect per crop, and merge results, rather than downscaling the whole frame at once.

Full annotated video output: `results/video/street_annotated.mp4`, generated via `python -m scripts.test_video` against a free, licensed street-scene clip (["A bustling day with pedestrians crossing a vibrant city street"](https://www.pexels.com/video/people-walking-on-the-street-3552510/) by Marc Van den Broeck, Pexels License). 

### Live-view latency

The `/live` browser view and the `/ws/detections` websocket both report a per-frame timing breakdown (`read_ms`, `detect_ms`, and `encode_ms` when frames are included). The pipeline is fully serial: read a frame, run inference, JPEG-encode it, send it, wait for the next request-response cycle to draw it, with no frame buffering or threading decoupling capture from inference. Perceived lag in `/live` shows real per-frame inference time (~150-250ms with the ONNX backend on CPU). A production system would decouple capture and inference with a frame queue to smooth this out, but in this project it's kept simple.

## Tracking

Added a lightweight IoU/centroid tracker on top of raw per-frame detection, which fixed the box-flicker effect found earlier. Instead of every frame being detected in isolation, `Tracker.update()` links detections across frames into persistent tracked boxes with their own `track_id`. The websocket payload and zone-alerting key off tracked, not raw, detections.

### Tracking algorithm

Tracking uses two rounds of greedy matching per frame, first by IoU (cheap, scale-aware but degrades to zero once two boxes stop overlapping), then by centroid-distance as a fallback for the leftovers. Centroid distance handles cases where motion ebtween frames outruns IoU's overlap requirements. Uses greedy assignment because for this project optimal-assignment cases are rare and not worth a dependency for the Hungarian algorithm.

Matching includes `min_hits`/`max_age` to measure each tracks state. A new detection starts as an unconfirmed tentative track and only becomes a reported and alertable identity after `min_hits` consecutive matches, which stops single frame false positives from getting an ID. A track that stops matching is held for `max_age` consecutive frames before it's dropped, allowing tracked boxes to survive some brief occlusion without losing the id entity.

### Tuning on testing footage

`iou_threshold` and `max_age` started at (0.3, 8) and were retuned after running `scripts/diagnose_tracking.py` and `scripts/diagnose_fragmentation.py`, which measured the real match-score and occlusion gap distribution for the ONNX detector output on each video, showing how the detector output jitters frame to frame. 

With the original `max_age=8`, 75.8% of all track deaths on video of a crowded street had a plausible same-person rebirth nearby within 1.5s/150px. Measured occlusion gap length (median 18 frames, 75th percentile 31, at 30fps), then set `max_age=30`, which bridges about 74% of observed gaps.
`iou_threshold` dropped from 0.3 to 0.15 after measuring near-miss IoU scores that the original default was silently rejecting. 

`centroid_max_dist` is scaled to 7% of the frame diagonal, and hardcoded alert zone was replaced with a similar resolution-scaled fix in `alerts.py`/`main.py`.

### Entryway demo

`results/video/entryway_tracked.mp4` (source: ["Delivery man delivering order"](https://www.pexels.com/video/delivery-man-delivering-order-6667223/) by Kampus Production, Pexels License). Track ID stays stable throughout the clip. Frames 41-47 (~0.28s) briefly show two overlapping tracks for the same person, due to detector producing two candidate boxes for one person with insufficient IoU overlap for NMS threshold to merge them. This self-corrects in 7 frames.

### Crowd demo

`results/video/street_tracked.mp4` is kept as a stress test. It's not what this project is made for, due to the busy crowd with many people obscuring eachother. Even after tuning, tight clusters of adjacent people still produce ID swaps when one briefly occludes another (e.g track `#7` -> `#23` mid-clip). This occurs because greedy IoU/centorid matching has no appearance signal to disambiguate which of several similarly scoring nearby candidates is the same person vs another person standing close by. To fix, you would use learned re-identification embedding (DeepSORT), which would add another model and latency.

## Benchmark

| variant | size (KB) | mean latency (ms) | p95 (ms) | fps |
|---|---|---|---|---|
| fp32 | 114.3 | 1.38 | 1.736 | 722.2 |
| int8 | 33.6 | 3.83 | 4.65 | 261.0 |

Size dropped 70%, but latency didn't improve. Dynamic quantization adds a dequantize step around every op, and on a small model that overhead is bigger than what you save doing the matmuls in int8. Thus, the effect of quantization on speed depends on the model being deep enough that the compute savings outweigh the added ops. Static quantization with a calibration set would probably behave differently, but I haven't tested that yet.

This benchmark runs on a small CNN built by hand with `onnx.helper`. To benchmark a real model, swap `MODEL_PATH` and bump `input_shape` to 640x640. Weights are seeded so the model itself is identical every run, but the absolute latency numbers move around depending on what else is running on the machine at the time, with swings of 2x+ between a quiet machine and a loaded one. The size reduction and the direction of the latency result (int8 slower, not faster) held steady across every run though.

HOG on its own runs about 154ms/frame (roughly 6.5fps) at 640px on a regular CPU, averaged over 20 runs after warmup.

### Same benchmark, on the real yolov8n

Once yolov8n.onnx was actually exported and the same FP32 vs INT8 comparison reran against it instead of the hand-built demo model, the result flipped:

| variant | size (KB) | mean latency (ms) | fps |
|---|---|---|---|
| fp32 | 12550 | 94.3 | 10.6 |
| int8 | 3422 | 65.6 | 15.2 |

INT8 is 1.44x faster here, because the dequant overhead is roughly fixed per op, but the compute it's saving scales with the model's actual size. On a network this small (a few thousand params) that overhead dominates, while on yolov8n (3.1M params, 8.7 GFLOPs) there's enough matmul work that quantizing it is a net win. A model's compute density is what determines whether quantization helps or hurts latency. See `results/benchmark_real_model.csv`.

However, the same comparison on different CPU hardware (AMD Ryzen 5 7520U vs. the machine above) came out the other way, int8 slightly slower (356ms vs 403ms fp32, 0.88x). Quantization's payoff depends on the specific CPU's int8 vs fp32 throughput characteristics, not just the model, could test this further (e.g on a Raspberry Pi).

## Architecture / design decisions

- **Two detectors behind one interface** (`app/detector.py`): `HogPersonDetector` (OpenCV's built-in HOG+SVM, zero setup) and `OnnxPersonDetector` (real YOLOv8 via onnxruntime), both returning the same `Detection` dataclass and `(detections, elapsed_time)` tuple, swappable via `DETECTOR_BACKEND` env var. A shared `draw_detections()` helper lives next to `Detection` so the box/label drawing style is defined once, not copy-pasted across `scripts/test_video.py`, `/demo/detect`, and anywhere else that needs it.
- **Drawing code lives in its own module** (`app/drawing.py`), not inside `detector.py`/`tracker.py`, because `draw_detections()` and `draw_tracks()` are the only things in the codebase that need `cv2`/`numpy` purely for visualization, so tracking's matching logic stays  testable in isolation.
- **Tracking is a lightweight IoU/centroid greedy matcher** not a heavier learned tracker.
- **Letterbox resize, not stretch-resize**, before feeding frames to the ONNX model the pipeline scales them to fit while preserving aspect ratio, pads the rest with grey, then undoes the scale+pad math on the output boxes. A straight resize would distort people's proportions and hurt accuracy. This is also the root cause of the dense-crowd failure mode above where the whole frame, including every tiny distant person, gets scaled down together.
- **NMS threshold of 0.45** for collapsing duplicate overlapping boxes for the same person, applied after class-filtering to person-only, vectorized with NumPy rather than a per-box Python loop (8400 candidate boxes per frame would otherwise dwarf the model's own inference time on CPU). This also causes the brief double detection on the entryway demo.
- **Redis for alert dedup**, via `SET NX EX` giving a 30-second cooldown per zone+label with no separate cleanup job needed, the key expires on its own.
- **Dependency injection via FastAPI's `Depends()`** (`app/dependencies.py`): the detector, Redis client, alert manager, and video frame source are all constructed through cached factory functions rather than bare module-level globals, so tests can cleanly swap in fakes (`fakeredis`, a stub detector, stub frame source) via `app.dependency_overrides` without touching route code. The `lifespan` context manager eagerly builds and pings these at startup (fail fast if the model file's missing or Redis is unreachable) rather than lazily on the first request. The frame source is not cached like the others, because each websocket connection needs its own `VideoCapture` handle.
- **Multi-stage Docker build**: a throwaway `builder` stage installs `ultralytics`/`torch` just long enough to export `yolov8n.onnx`, then the actual runtime image only inherits that one resulting file via `COPY --from=builder`. `torch` never ships in what's actually deployed.
- **`POST /demo/detect`** returns either an annotated JPEG or a JSON detection list (`format` query param), so it works both as a quick visual check and as a properly testable API from `/docs`. This is also the only endpoint that works against the live deployment with zero local setup, since it doesn't need a webcam or a persistent connection.

## Tests

30 tests, `pip install pytest fakeredis && pytest tests/ -v`:
- `test_detector.py`: NMS (suppression, survival, empty input, partial overlap below threshold)
- `test_tracker.py`: track confirmation gating (`min_hits`), surviving a one-frame gap under the same ID (the actual flicker fix, proven directly), expiry after `max_age` consecutive misses, two well-separated tracks not swapping IDs, and a track's label/confidence reflecting the real matched detection rather than a placeholder
- `test_alerts.py`: `Zone.overlaps_box`'s four separation directions plus edge-touching, `AlertManager`'s cooldown/dedup logic via `fakeredis`
- `test_main.py`: FastAPI route coverage (`/health`, `/alerts/recent`) via `TestClient`, plus a full end-to-end integration test (`test_alert_full_pipeline`) that fakes a detection and a frame source to prove a real detection crossing the zone actually fires an alert through the live websocket route (across enough frames for the track to confirm) and lands in `/alerts/recent` afterward

## Running it

```
docker compose up --build
curl http://localhost:8000/health
```

For a real webcam it's easier to skip Docker and just run it locally. Needs Redis installed or running via `docker run -d -p 6379:6379 redis:7-alpine`:

```
pip install -r requirements.txt
uvicorn app.main:app --reload
```
then go to `http://localhost:8000/live` in a browser with `VIDEO_SOURCE=0` set, or connect a websocket client directly to `ws://localhost:8000/ws/detections`.

For the ONNX backend outside Docker:
```
pip install ultralytics
cd models && yolo export model=yolov8n.pt format=onnx imgsz=640
```
(Docker handles this automatically via the multi-stage build, this is only needed for running the export locally outside a container.)

### Deployed version

The live Render deployment runs the ONNX backend (`yolov8n.onnx`, bundled into the image at build time) with a managed Redis instance, provisioned via `render.yaml` as a Blueprint. No `VIDEO_SOURCE` is set (no camera on a cloud server), so `/health`, `/docs`, `/demo`, `/demo/detect`, `/alerts/recent`, and `/benchmark/results` all work, but `/live` and `/ws/detections` have nothing to stream from there.

## What this isn't

This is a benchmarking harness and an alerting service, not an on-device deployment. It runs on a regular CPU through onnxruntime and doesn't target any specific camera SoC, NPU, or cross-compiled runtime. It's also not a crowd-counting system. Dense-crowd scenes are a known failure mode.

## TODO / known gaps

- HOG is not very accurate, it's there because it needs no download. I would use the ONNX path for anything real
- accuracy eval is on a 40-image sample. This is worth widening for tighter confidence intervals
- no auth on the websocket or `/demo/detect`
- alert history is just whatever's in Redis's bounded list, nothing persisted long term
- single camera only right now
- zone is still a computed default (`default_zone_for_resolution`), not yet user-configurable per camera -- correct across resolutions now, but a real deployment would want this drawn by a user in a setup UI, not any default at all
- want to try static quantization + FP16 and see if the int8-helps result holds up further
- dense-crowd scenes are still alimitation for a motion-only tracker, to fix would add re-identification embedding.
- tiled/sliding-window inference for dense-crowd *detection* (as opposed to tracking) also not yet tried