#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Generate single-op ONNX fixtures for the WS4 op matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install with: pip install onnx numpy"
    ) from exc

SEED_OPS = ("add", "mul", "matmul", "relu", "reshape")
EXTENDED_OPS = (
    "sub",
    "div",
    "sigmoid",
    "reduce_mean",
    "reduce_sum",
    "transpose",
    "concat",
    "slice",
    "conv",
    "layer_norm",
    "softmax",
    "gather",
)
FULL_OPS = SEED_OPS + EXTENDED_OPS


def _model(
    nodes: list,
    inputs: list,
    outputs: list,
    initializers: list | None = None,
    name: str = "op_graph",
    opset: int = 13,
) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        name,
        inputs,
        outputs,
        initializer=initializers or [],
    )
    model = helper.make_model(
        graph,
        producer_name="tt-xla-onnx-op-matrix",
        opset_imports=[helper.make_opsetid("", opset)],
    )
    onnx.checker.check_model(model)
    return model


def build_add_model() -> onnx.ModelProto:
    input_a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 4])
    input_b = helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, 4])
    output = helper.make_tensor_value_info("C", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Add", ["A", "B"], ["C"], name="add0")
    return _model([node], [input_a, input_b], [output], name="add_graph")


def build_mul_model() -> onnx.ModelProto:
    input_a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 4])
    input_b = helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, 4])
    output = helper.make_tensor_value_info("C", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Mul", ["A", "B"], ["C"], name="mul0")
    return _model([node], [input_a, input_b], [output], name="mul_graph")


def build_matmul_model() -> onnx.ModelProto:
    input_a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 3])
    input_b = helper.make_tensor_value_info("B", TensorProto.FLOAT, [3, 4])
    output = helper.make_tensor_value_info("C", TensorProto.FLOAT, [2, 4])
    node = helper.make_node("MatMul", ["A", "B"], ["C"], name="matmul0")
    return _model([node], [input_a, input_b], [output], name="matmul_graph")


def build_relu_model() -> onnx.ModelProto:
    input_x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    output_y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Relu", ["X"], ["Y"], name="relu0")
    return _model([node], [input_x], [output_y], name="relu_graph")


def build_reshape_model() -> onnx.ModelProto:
    input_x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    input_shape = helper.make_tensor_value_info("shape", TensorProto.INT64, [2])
    output_y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 2])
    node = helper.make_node("Reshape", ["X", "shape"], ["Y"], name="reshape0")
    return _model([node], [input_x, input_shape], [output_y], name="reshape_graph")


def build_sub_model() -> onnx.ModelProto:
    input_a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 4])
    input_b = helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, 4])
    output = helper.make_tensor_value_info("C", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Sub", ["A", "B"], ["C"], name="sub0")
    return _model([node], [input_a, input_b], [output], name="sub_graph")


def build_div_model() -> onnx.ModelProto:
    input_a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 4])
    input_b = helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, 4])
    output = helper.make_tensor_value_info("C", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Div", ["A", "B"], ["C"], name="div0")
    return _model([node], [input_a, input_b], [output], name="div_graph")


def build_sigmoid_model() -> onnx.ModelProto:
    input_x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    output_y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Sigmoid", ["X"], ["Y"], name="sigmoid0")
    return _model([node], [input_x], [output_y], name="sigmoid_graph")


def build_reduce_mean_model() -> onnx.ModelProto:
    input_x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    output_y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1])
    # ReduceMean-13: axes remains an attribute (unlike ReduceSum-13).
    node = helper.make_node(
        "ReduceMean",
        ["X"],
        ["Y"],
        name="reduce_mean0",
        axes=[1],
        keepdims=1,
    )
    return _model([node], [input_x], [output_y], name="reduce_mean_graph")


def build_reduce_sum_model() -> onnx.ModelProto:
    input_x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    output_y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1])
    # Static axes attribute (opset 11) — avoids onnx-mlir shape-dialect axes input
    # that can leak shape.get_extent into StableHLO (see onnx_implementation_log).
    node = helper.make_node(
        "ReduceSum",
        ["X"],
        ["Y"],
        name="reduce_sum0",
        axes=[1],
        keepdims=1,
    )
    return _model(
        [node],
        [input_x],
        [output_y],
        name="reduce_sum_graph",
        opset=11,
    )


def build_transpose_model() -> onnx.ModelProto:
    input_x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
    output_y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 2])
    node = helper.make_node("Transpose", ["X"], ["Y"], name="transpose0", perm=[1, 0])
    return _model([node], [input_x], [output_y], name="transpose_graph")


def build_concat_model() -> onnx.ModelProto:
    input_a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 2])
    input_b = helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, 2])
    output = helper.make_tensor_value_info("C", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Concat", ["A", "B"], ["C"], name="concat0", axis=1)
    return _model([node], [input_a, input_b], [output], name="concat_graph")


def build_slice_model() -> onnx.ModelProto:
    input_x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    output_y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])
    # Static slice params (initializers) — onnx-mlir requires compile-time axes.
    starts = numpy_helper.from_array(np.array([1], dtype=np.int64), name="starts")
    ends = numpy_helper.from_array(np.array([3], dtype=np.int64), name="ends")
    axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name="axes")
    steps = numpy_helper.from_array(np.array([1], dtype=np.int64), name="steps")
    node = helper.make_node(
        "Slice",
        ["X", "starts", "ends", "axes", "steps"],
        ["Y"],
        name="slice0",
    )
    return _model(
        [node],
        [input_x],
        [output_y],
        initializers=[starts, ends, axes, steps],
        name="slice_graph",
    )


def build_conv_model() -> onnx.ModelProto:
    input_x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 8, 8])
    output_y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 8, 8])
    weight = numpy_helper.from_array(
        np.ones((1, 1, 3, 3), dtype=np.float32) / 9.0,
        name="W",
    )
    # Explicit zero bias — omitting bias makes onnx-mlir emit onnx.NoValue in SHLO.
    bias = numpy_helper.from_array(np.zeros((1,), dtype=np.float32), name="B")
    node = helper.make_node(
        "Conv",
        ["X", "W", "B"],
        ["Y"],
        name="conv0",
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
    )
    return _model(
        [node],
        [input_x],
        [output_y],
        initializers=[weight, bias],
        name="conv_graph",
    )


def build_layer_norm_model() -> onnx.ModelProto:
    """Decomposed LayerNorm (M1.8 allows decomposed form; onnx-mlir SHLO gap on op)."""
    eps = 1e-5
    input_x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])
    input_scale = helper.make_tensor_value_info("scale", TensorProto.FLOAT, [4])
    input_bias = helper.make_tensor_value_info("bias", TensorProto.FLOAT, [4])
    output_y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])
    eps_init = numpy_helper.from_array(np.array(eps, dtype=np.float32), name="eps")
    nodes = [
        helper.make_node(
            "ReduceMean", ["X"], ["mean"], name="mean", axes=[1], keepdims=1
        ),
        helper.make_node("Sub", ["X", "mean"], ["centered"], name="sub"),
        helper.make_node("Mul", ["centered", "centered"], ["sq"], name="sq"),
        helper.make_node(
            "ReduceMean", ["sq"], ["var"], name="var", axes=[1], keepdims=1
        ),
        helper.make_node("Add", ["var", "eps"], ["var_eps"], name="var_eps"),
        helper.make_node("Sqrt", ["var_eps"], ["std"], name="std"),
        helper.make_node("Div", ["centered", "std"], ["norm"], name="norm"),
        helper.make_node("Mul", ["norm", "scale"], ["scaled"], name="scale"),
        helper.make_node("Add", ["scaled", "bias"], ["Y"], name="bias"),
    ]
    return _model(
        nodes,
        [input_x, input_scale, input_bias],
        [output_y],
        initializers=[eps_init],
        name="layer_norm_graph",
        opset=13,
    )


def build_softmax_model() -> onnx.ModelProto:
    """Decomposed Softmax on axis=1 (onnx-mlir stablehlo gap on single Softmax op)."""
    input_x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    output_y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
    nodes = [
        helper.make_node(
            "ReduceMax", ["X"], ["xmax"], name="xmax", axes=[1], keepdims=1
        ),
        helper.make_node("Sub", ["X", "xmax"], ["shifted"], name="shift"),
        helper.make_node("Exp", ["shifted"], ["exp"], name="exp"),
        helper.make_node(
            "ReduceSum", ["exp"], ["sum"], name="sum", axes=[1], keepdims=1
        ),
        helper.make_node("Div", ["exp", "sum"], ["Y"], name="softmax"),
    ]
    # Opset 11: ReduceSum/ReduceMax use static axes attributes (opset 13+ moves
    # ReduceSum axes to an input, which onnx-mlir lowers with shape dialect).
    return _model(nodes, [input_x], [output_y], name="softmax_graph", opset=11)


def build_gather_model() -> onnx.ModelProto:
    input_data = helper.make_tensor_value_info("data", TensorProto.FLOAT, [3, 4])
    input_indices = helper.make_tensor_value_info("indices", TensorProto.INT64, [2])
    output_y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])
    node = helper.make_node("Gather", ["data", "indices"], ["Y"], name="gather0", axis=0)
    return _model([node], [input_data, input_indices], [output_y], name="gather_graph")


_BUILDERS = {
    "add": build_add_model,
    "mul": build_mul_model,
    "matmul": build_matmul_model,
    "relu": build_relu_model,
    "reshape": build_reshape_model,
    "sub": build_sub_model,
    "div": build_div_model,
    "sigmoid": build_sigmoid_model,
    "reduce_mean": build_reduce_mean_model,
    "reduce_sum": build_reduce_sum_model,
    "transpose": build_transpose_model,
    "concat": build_concat_model,
    "slice": build_slice_model,
    "conv": build_conv_model,
    "layer_norm": build_layer_norm_model,
    "softmax": build_softmax_model,
    "gather": build_gather_model,
}


def default_feed(op: str) -> dict[str, list[Any]]:
    rng = np.random.default_rng(0)
    if op == "add":
        return {
            "A": [[1.0, 2.0, 3.0, 4.0]],
            "B": [[4.0, 3.0, 2.0, 1.0]],
        }
    if op == "mul":
        return {
            "A": [[1.0, 2.0, 3.0, 4.0]],
            "B": [[4.0, 3.0, 2.0, 1.0]],
        }
    if op == "matmul":
        return {
            "A": rng.standard_normal((2, 3)).astype(np.float32).tolist(),
            "B": rng.standard_normal((3, 4)).astype(np.float32).tolist(),
        }
    if op == "relu":
        return {"X": [[-1.0, 0.0, 1.0, 2.0]]}
    if op == "reshape":
        return {"X": [[1.0, 2.0, 3.0, 4.0]], "shape": [2, 2]}
    if op == "sub":
        return {
            "A": [[4.0, 3.0, 2.0, 1.0]],
            "B": [[1.0, 2.0, 3.0, 4.0]],
        }
    if op == "div":
        return {
            "A": [[4.0, 6.0, 8.0, 10.0]],
            "B": [[2.0, 3.0, 4.0, 5.0]],
        }
    if op == "sigmoid":
        return {"X": [[-2.0, -1.0, 0.0, 1.0]]}
    if op == "reduce_mean":
        return {"X": [[1.0, 2.0, 3.0, 4.0]]}
    if op == "reduce_sum":
        return {"X": [[1.0, 2.0, 3.0, 4.0]]}
    if op == "transpose":
        return {"X": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]}
    if op == "concat":
        return {
            "A": [[1.0, 2.0]],
            "B": [[3.0, 4.0]],
        }
    if op == "slice":
        return {"X": [[1.0, 2.0, 3.0, 4.0]]}
    if op == "conv":
        return {"X": rng.standard_normal((1, 1, 8, 8)).astype(np.float32).tolist()}
    if op == "layer_norm":
        return {
            "X": rng.standard_normal((2, 4)).astype(np.float32).tolist(),
            "scale": [1.0, 1.0, 1.0, 1.0],
            "bias": [0.0, 0.0, 0.0, 0.0],
        }
    if op == "softmax":
        return {"X": [[1.0, 2.0, 3.0, 4.0]]}
    if op == "gather":
        return {
            "data": [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ],
            "indices": [0, 2],
        }
    raise KeyError(f"Unknown op: {op}")


def build_model(op: str) -> onnx.ModelProto:
    try:
        return _BUILDERS[op]()
    except KeyError as exc:
        raise SystemExit(
            f"Unknown op {op!r}. Choose from: {', '.join(FULL_OPS)}"
        ) from exc


def op_list(*, full: bool) -> tuple[str, ...]:
    return FULL_OPS if full else SEED_OPS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "op",
        nargs="?",
        choices=FULL_OPS,
        help="Op to generate (omit with --all)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output .onnx path (default: fixtures/<op>.onnx)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all seed op fixtures (use with --full for M1.8 matrix)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Include M1.8 extended ops (17 total with --all)",
    )
    parser.add_argument(
        "--write-feeds",
        type=Path,
        help="Write default input feeds JSON for the generated op(s)",
    )
    args = parser.parse_args()

    fixture_dir = Path(__file__).resolve().parent / "fixtures"
    ops = list(op_list(full=args.full)) if args.all else [args.op]
    if not ops or ops == [None]:
        parser.error("Provide op name or --all")

    feeds: dict[str, dict[str, list[Any]]] = {}
    for op in ops:
        out = args.output if args.output and not args.all else fixture_dir / f"{op}.onnx"
        out.parent.mkdir(parents=True, exist_ok=True)
        model = build_model(op)
        onnx.save(model, out)
        feeds[op] = default_feed(op)
        print(f"Wrote {out}")

    if args.write_feeds:
        args.write_feeds.parent.mkdir(parents=True, exist_ok=True)
        payload = feeds if args.all else {ops[0]: feeds[ops[0]]}
        args.write_feeds.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.write_feeds}")


if __name__ == "__main__":
    main()
