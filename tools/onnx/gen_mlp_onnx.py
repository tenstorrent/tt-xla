#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Generate a tiny 2-layer MLP ONNX model for WS4 e2e tests.

Uses elementwise Mul/Add/Relu (no Gemm). ONNX Gemm lowers to ``stablehlo.dot``,
which tt-mlir currently fails to legalize to TTIR — see ``gen_mlp_gemm_onnx.py``
for that variant and ``onnx_implementation_log.md`` WS4 blockers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install with: pip install onnx numpy"
    ) from exc


def build_mlp_model(seed: int = 0) -> onnx.ModelProto:
    """2-layer per-element MLP: Mul → Add → Relu → Mul → Add."""
    rng = np.random.default_rng(seed)
    scale1 = rng.standard_normal((1, 4), dtype=np.float32) * 0.25 + 1.0
    bias1 = rng.standard_normal((1, 4), dtype=np.float32) * 0.1
    scale2 = rng.standard_normal((1, 4), dtype=np.float32) * 0.25 + 1.0
    bias2 = rng.standard_normal((1, 4), dtype=np.float32) * 0.1

    input_x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    output_y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])

    init_scale1 = numpy_helper.from_array(scale1, name="scale1")
    init_bias1 = numpy_helper.from_array(bias1, name="bias1")
    init_scale2 = numpy_helper.from_array(scale2, name="scale2")
    init_bias2 = numpy_helper.from_array(bias2, name="bias2")

    mul1 = helper.make_node("Mul", inputs=["X", "scale1"], outputs=["M1"], name="mul1")
    add1 = helper.make_node("Add", inputs=["M1", "bias1"], outputs=["H1"], name="add1")
    relu1 = helper.make_node("Relu", inputs=["H1"], outputs=["H2"], name="relu1")
    mul2 = helper.make_node("Mul", inputs=["H2", "scale2"], outputs=["M2"], name="mul2")
    add2 = helper.make_node("Add", inputs=["M2", "bias2"], outputs=["Y"], name="add2")

    graph = helper.make_graph(
        [mul1, add1, relu1, mul2, add2],
        "mlp_elem_graph",
        [input_x],
        [output_y],
        initializer=[init_scale1, init_bias1, init_scale2, init_bias2],
    )
    model = helper.make_model(
        graph,
        producer_name="tt-xla-onnx-ws4",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "fixtures" / "mlp2.onnx",
        help="Output .onnx path",
    )
    parser.add_argument("--seed", type=int, default=0, help="Weight RNG seed")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = build_mlp_model(seed=args.seed)
    onnx.save(model, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
