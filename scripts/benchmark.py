"""FP32 vs INT8 benchmark: size, latency, throughput.

    python scripts/benchmark.py
    python scripts/benchmark.py --model models/yolov8n.onnx --input-size 640 --out results/benchmark_real_model.csv
"""

import argparse
import csv
import os
import time

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

NUM_WARMUP = 5
NUM_RUNS = 50


def quantize(fp32_path: str, int8_path: str) -> None:
    quantize_dynamic(
        model_input=fp32_path,
        model_output=int8_path,
        weight_type=QuantType.QUInt8,
    )


def benchmark_model(model_path: str, input_shape: tuple) -> dict:
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # Warmup: first few runs pay one-time costs (memory allocation, thread
    # pool spin-up) that would skew latency numbers if included.
    for _ in range(NUM_WARMUP):
        session.run(None, {input_name: dummy_input})

    latencies = []
    t_start = time.perf_counter()
    for _ in range(NUM_RUNS):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        latencies.append(time.perf_counter() - t0)
    total_time = time.perf_counter() - t_start

    latencies = np.array(latencies) * 1000  # -> ms
    return {
        "file_size_kb": os.path.getsize(model_path) / 1024,
        "mean_latency_ms": float(np.mean(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "throughput_fps": NUM_RUNS / total_time,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/demo_fp32.onnx",
                         help="Path to the FP32 model to benchmark (default: the demo model).")
    parser.add_argument("--input-size", type=int, default=224,
                         help="Square input size in pixels: 224 for the demo model, 640 for yolov8n.")
    parser.add_argument("--out", default="results/benchmark.csv",
                         help="Where to write the results CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model
    quantized_path = model_path.replace(".onnx", "_int8.onnx")

    if not os.path.exists(model_path):
        raise SystemExit(
            f"No model at {model_path}. Run scripts/make_demo_model.py first "
            f"(or pass --model pointing at your own exported model)."
        )

    print(f"Quantizing {model_path} -> {quantized_path} ...")
    quantize(model_path, quantized_path)

    input_shape = (1, 3, args.input_size, args.input_size)

    print("Benchmarking FP32...")
    fp32_results = benchmark_model(model_path, input_shape)

    print("Benchmarking INT8...")
    int8_results = benchmark_model(quantized_path, input_shape)

    rows = [
        {"variant": "fp32", **fp32_results},
        {"variant": "int8", **int8_results},
    ]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_path = args.out
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults written to {out_path}\n")
    for r in rows:
        print(f"{r['variant']}: {r['file_size_kb']:.1f} KB, "
              f"{r['mean_latency_ms']:.3f} ms mean, "
              f"{r['p95_latency_ms']:.3f} ms p95, "
              f"{r['throughput_fps']:.1f} fps")

    size_reduction = (1 - int8_results["file_size_kb"] / fp32_results["file_size_kb"]) * 100
    speedup = fp32_results["mean_latency_ms"] / int8_results["mean_latency_ms"]
    print(f"\nSize reduction: {size_reduction:.1f}%")
    print(f"Latency ratio (fp32/int8): {speedup:.2f}x")


if __name__ == "__main__":
    main()