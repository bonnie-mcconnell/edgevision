"""FP32 vs INT8 benchmark: size, latency, throughput.

    python scripts/benchmark.py
    python scripts/benchmark.py --model models/yolov8n.onnx --input-size 640 --out results/benchmark_real_model.csv
"""

import argparse
import csv
import os
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnxconverter_common import float16
from onnxruntime.quantization import QuantType, quantize_dynamic, quantize_static, CalibrationDataReader
from onnxruntime.quantization.preprocess import quant_pre_process

NUM_WARMUP = 5
NUM_RUNS = 50


def quantize_dynamic_int8(fp32_path: str, int8_path: str) -> None:
    quantize_dynamic(
        model_input=fp32_path,
        model_output=int8_path,
        weight_type=QuantType.QUInt8,
    )


def convert_to_fp16(fp32_path: str, fp16_path: str) -> None:
    """
    Halves weight storage converting from float32 -> float16.
    No calibration needed because it's a per-value cast. 
    Effect on latency depends on the CPU running it: if it has a
    fast native fp16 compute path or if onnxruntime upcasts back to
    fp32 internally to run operations.
    """
    model = onnx.load(fp32_path)
    model_fp16 = float16.convert_float_to_float16(model, op_block_list=["Resize", "Concat"])
    onnx.save(model_fp16, fp16_path)


def preprocess_for_static_quant(fp32_path: str, preprocessed_path: str) -> None:
    """
    quantize_static() warns without this because it needs shape inference
    run over the graph first to place quantization nodes correctly, 
    especially around Conv layers.
    """
    quant_pre_process(fp32_path, preprocessed_path)


def benchmark_model(model_path: str, input_shape: tuple) -> dict:
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    dtype = np.float16 if "float16" in input_info.type else np.float32
    dummy_input = np.random.randn(*input_shape).astype(dtype)

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


class RandomCalibrationDataReader(CalibrationDataReader):
    """
    Feeds quantize_static() synthetic Gaussian noise as calibration data,
    used here because this benchmark measures latency/size not accuracy.
    Calibration sets the clipping ranges used for quantization scale/zero-point,
    doesn't change the op graph so doesn't affect what is being measured.
    Do not use this to validate quantized-model accuracy.
    """
    def __init__(self, input_name: str, input_shape: tuple, num_samples: int = 20):
        self.input_name = input_name
        self.input_shape = input_shape
        self.num_samples = num_samples
        self.count = 0

    def get_next(self) -> dict | None:
        """Called by quantize_static() repeatedly until it returns None.
        Every non-None return is one calibration sample: {input_name: array}."""
        if self.count >= self.num_samples:
            return None
        self.count += 1
        return {self.input_name: np.random.randn(*self.input_shape).astype(np.float32)}


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

    if not os.path.exists(model_path):
        raise SystemExit(
            f"No model at {model_path}. Run scripts/make_demo_model.py first "
            f"(or pass --model pointing at your own exported model)."
        )

    input_shape = (1, 3, args.input_size, args.input_size)
    base_name = model_path.replace(".onnx", "")

    int8_dynamic_path = f"{base_name}_int8_dynamic.onnx"
    preprocessed_path = f"{base_name}_preprocessed.onnx"
    int8_static_path = f"{base_name}_int8_static.onnx"
    fp16_path = f"{base_name}_fp16.onnx"

    print(f"Quantizing (dynamic) {model_path} -> {int8_dynamic_path} ...")
    quantize_dynamic_int8(model_path, int8_dynamic_path)

    print(f"Preprocessing {model_path} for static quant -> {preprocessed_path} ...")
    preprocess_for_static_quant(model_path, preprocessed_path)

    # calibration reader needs graph's real input name
    input_name = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"]).get_inputs()[0].name
    calibration_reader = RandomCalibrationDataReader(input_name, input_shape)

    print(f"Quantizing (static) {preprocessed_path} -> {int8_static_path} ...")
    quantize_static(
        model_input=preprocessed_path,
        model_output=int8_static_path,
        calibration_data_reader=calibration_reader,
    )

    print(f"Converting to fp16 -> {fp16_path} ...")
    convert_to_fp16(model_path, fp16_path)

    variants = [
        ("fp32", model_path),
        ("int8_dynamic", int8_dynamic_path),
        ("int8_static", int8_static_path),
        ("fp16", fp16_path),
    ]

    rows = []
    for name, path in variants:
        print(f"Benchmarking {name}...")
        result = benchmark_model(path, input_shape)
        rows.append({"variant": name, **result})

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults written to {args.out}\n")
    for r in rows:
        print(f"{r['variant']}: {r['file_size_kb']:.1f} KB, "
              f"{r['mean_latency_ms']:.3f} ms mean, "
              f"{r['p95_latency_ms']:.3f} ms p95, "
              f"{r['throughput_fps']:.1f} fps")

    fp32 = rows[0]
    print()
    for r in rows[1:]:
        size_reduction = (1 - r["file_size_kb"] / fp32["file_size_kb"]) * 100
        speedup = fp32["mean_latency_ms"] / r["mean_latency_ms"]
        print(f"{r['variant']} vs fp32: {size_reduction:+.1f}% size, {speedup:.2f}x latency")


if __name__ == "__main__":
    main()