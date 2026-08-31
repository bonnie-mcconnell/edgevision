"""
Small CNN built as raw ONNX ops instead of exported from torch, because I needed
something real for the quantize/benchmark scripts to run against without
pulling in torch + a full pretrained model.
"""

from __future__ import annotations

import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def conv_block(
    name: str, in_channels: int, out_channels: int, input_name: str
) -> tuple[list, list, str]:
    """One Conv -> ReLU block. Returns (nodes, initializers, output_name)."""
    weight = np.random.randn(out_channels, in_channels, 3, 3).astype(np.float32) * 0.1
    bias = np.zeros(out_channels, dtype=np.float32)

    w_init = numpy_helper.from_array(weight, name=f"{name}_w")
    b_init = numpy_helper.from_array(bias, name=f"{name}_b")

    conv_out = f"{name}_conv_out"
    relu_out = f"{name}_relu_out"

    conv_node = helper.make_node(
        "Conv", [input_name, f"{name}_w", f"{name}_b"], [conv_out],
        kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[2, 2],
    )
    relu_node = helper.make_node("Relu", [conv_out], [relu_out])

    return [conv_node, relu_node], [w_init, b_init], relu_out


def build_model(path: str) -> None:
    input_name = "input"
    channels = [3, 16, 32, 64]

    all_nodes, all_inits = [], []
    current = input_name
    for i in range(len(channels) - 1):
        nodes, inits, current = conv_block(f"block{i}", channels[i], channels[i + 1], current)
        all_nodes += nodes
        all_inits += inits

    # Global average pool + flatten + a linear "detection head" so the
    # output shape resembles a real model's raw prediction tensor
    pool_out = "pool_out"
    all_nodes.append(helper.make_node("GlobalAveragePool", [current], [pool_out]))

    flat_out = "flat_out"
    all_nodes.append(helper.make_node("Flatten", [pool_out], [flat_out]))

    fc_weight = np.random.randn(64, 84).astype(np.float32) * 0.1
    fc_bias = np.zeros(84, dtype=np.float32)
    all_inits.append(numpy_helper.from_array(fc_weight, name="fc_w"))
    all_inits.append(numpy_helper.from_array(fc_bias, name="fc_b"))
    all_nodes.append(helper.make_node("Gemm", [flat_out, "fc_w", "fc_b"], ["output"]))

    graph = helper.make_graph(
        all_nodes,
        "edge_detector_demo",
        [helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1, 3, 224, 224])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 84])],
        all_inits,
    )

    model = helper.make_model(graph, producer_name="edgevision-demo")
    model.opset_import[0].version = 13
    onnx.checker.check_model(model)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    onnx.save(model, path)
    print(f"wrote {path}")


if __name__ == "__main__":
    np.random.seed(0)  # reproducible weights, so the benchmark numbers don't drift between runs
    build_model("models/demo_fp32.onnx")
