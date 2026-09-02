# EdgeVision

Person detection + zone alerting for a security camera use case, with a benchmark comparing FP32 vs INT8 quantized inference on CPU.

Video frame arrives, a detector finds people, boxes get checked against a zone, and alerts get deduped through Redis before hitting a websocket. There's also a separate benchmark script for testing whether quantizing a model helps on CPU.

## Benchmark

| variant | size (KB) | mean latency (ms) | p95 (ms) | fps |
|---|---|---|---|---|
| fp32 | 114.3 | 1.38 | 1.736 | 722.2 |
| int8 | 33.6 | 3.83 | 4.65 | 261.0 |

Size dropped 70% like you'd expect. Latency didn't improve though, it got worse. Dynamic quantization adds a dequantize step around every op and on a small model that overhead is bigger than what you save doing the matmuls in int8. So "quantize it, it'll be faster" isn't true in general, it depends on the model being deep enough that the compute savings outweigh the added ops. Static quantization with a calibration set would probably behave differently, but I haven't tested that yet.

This benchmark runs on a tiny CNN I built by hand with onnx.helper instead of a real yolov8n, mainly because pulling in torch + a full pretrained model felt like overkill for testing whether the quantize/benchmark code works. The methodology's the same: swap MODEL_PATH once you've exported a real one and bump input_shape to 640x640. Weights are seeded so the model itself is identical every run, but the absolute latency numbers move around depending on what else is running on the machine at the time, I've seen swings of 2x+ between a quiet machine and a loaded one. The size reduction and the direction of the latency result (int8 slower, not faster) held steady across every run though.

HOG on its own runs about 154ms/frame (roughly 6.5fps) at 640px on a regular CPU, averaged over 20 runs after warmup, for reference.

### Same benchmark, on the real yolov8n

Once I actually exported yolov8n.onnx (see below) and reran the same FP32 vs INT8 comparison against it instead of the hand-built demo model, the result flipped:

| variant | size (KB) | mean latency (ms) | fps |
|---|---|---|---|
| fp32 | 12550 | 94.3 | 10.6 |
| int8 | 3422 | 65.6 | 15.2 |

INT8 is 1.44x faster here, because the dequant overhead is roughly fixed per op, but the compute it's saving scales with the model's actual size. On a network this small (a few thousand params) that overhead dominates; on yolov8n (3.1M params, 8.7 GFLOPs) there's enough real matmul work that quantizing it is a net win, meaning that the model's compute density that determines the impact quantization has on latency. See `results/benchmark_real_model.csv`.

## What's in here

`app/detector.py` has two detectors under the same interface: `HogPersonDetector` (OpenCV's built-in HOG+SVM, zero setup, this is what runs by default) and `OnnxPersonDetector` for a real YOLOv8 model:
```
pip install ultralytics
cd models && yolo export model=yolov8n.pt format=onnx imgsz=640
```
`app/alerts.py` does the zone check + Redis dedup so a person standing around for 10 seconds doesn't spam 300 alerts, `app/main.py` is the FastAPI/websocket layer, `scripts/make_demo_model.py` builds a small ONNX model by hand so there's something real to benchmark against, `scripts/benchmark.py` runs the FP32 vs INT8 comparison.

### Accuracy check on real images

Ran the exported yolov8n.onnx through `OnnxPersonDetector` against two standard CV test images (bundled with the `ultralytics` package, so no network fetch needed):

- `bus.jpg`: 4 people in frame, found all 4 (conf 0.89, 0.88, 0.88, 0.44 - the low-confidence one is partially occluded behind the bus door, correctly still caught).
- `zidane.jpg`: 2 people in frame, found both at 0.83 each.

This isn't a formal precision/recall eval against a labeled dataset (that's still a TODO), but it's some evidence the pipeline actually detects real people correctly end-to-end.

Accuracy check with multiple confidence thresholds:

conf	Precision	Recall	tp	fp	fn
0.25	0.797	0.615	110	28	69
0.40	0.877	0.559	100	14	79
0.50	0.917	0.492	88	8	91
0.60	0.974	0.413	74	2	105

`coco_subset/` is gitignored (regenerate with `python scripts/fetch_coco_subset.py`, seeded for reproducibility).

### Tests

`tests/` covers the pure-logic pieces that are cheap to test and easy to get subtly wrong: `Zone.overlaps_box` (all four separation cases plus edge-touching), NMS (suppression, survival, empty input), and `AlertManager`'s cooldown/dedup logic (via `fakeredis`, no real Redis needed to run the suite). `pip install pytest fakeredis && pytest tests/ -v`.

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

then connect to `ws://localhost:8000/ws/detections`.

The Dockerfile exports `yolov8n.onnx` at build time via a multi-stage build (a throwaway builder stage installs `ultralytics`/`torch` just long enough to run the export, then the final runtime image only inherits the resulting `.onnx` file, `torch` never ships in what's actually deployed). Both `docker-compose.yml` and the live Render deployment run the ONNX backend as a result, no manual model upload or volume mount needed.

## What this isn't

This is a benchmarking harness and an alerting service, not an on-device deployment. It runs on a regular CPU through onnxruntime and doesn't target any specific camera SoC, NPU, or cross-compiled runtime, that's a different/bigger problem than what's built here.

## TODO / known gaps

- HOG is not very accurate, it's there because it needs no download. would swap to the ONNX path for anything real
- no formal precision/recall eval against a labeled dataset yet, just spot-checked against two known test images
- no auth on the websocket
- alert history is just whatever's in Redis's bounded list, nothing persisted long term
- single camera only right now
- want to try static quantization + FP16 and see if the int8-helps result holds up further