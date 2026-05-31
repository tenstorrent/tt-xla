#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Generate Gemm-based 2-layer MLP ONNX (expected tt-mlir compile failure).

ONNX Gemm → ``stablehlo.dot`` → SHLO→TTIR fails today with:
  ``failed to legalize operation 'stablehlo.dot'``

Kept for gap tracking; use ``gen_mlp_onnx.py`` for WS4 e2e that runs on device.
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


def build_mlp_gemm_model(seed: int = 0) -> onnx.ModelProto:
    rng = np.random.default_rng(seed)
    w1 = rng.standard_normal((4, 8), dtype=np.float32) * 0.1
    b1 = rng.standard_normal((8,), dtype=np.float32) * 0.1
    w2 = rng.standard_normal((8, 4), dtype=np.float32) * 0.1
    b2 = rng.standard_normal((4,), dtype=np.float32) * 0.1

    input_x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    output_y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])

    gemm1 = helper.make_node(
        "Gemm",
        ["X", "W1", "b1"],
        ["H1"],
        name="gemm1",
        alpha=1.0,
        beta=1.0,
        transA=0,
        transB=0,
    )
    relu1 = helper.make_node("Relu", ["H1"], ["H2"], name="relu1")
    gemm2 = helper.make_node(
        "Gemm",
        ["H2", "W2", "b2"],
        ["Y"],
        name="gemm2",
        alpha=1.0,
        beta=1.0,
        transA=0,
        transB=0,
    )

    graph = helper.make_graph(
        [gemm1, relu1, gemm2],
        "mlp_gemm_graph",
        [input_x],
        [output_y],
        initializer=[
            numpy_helper.from_array(w1, name="W1"),
            numpy_helper.from_array(b1, name="b1"),
            numpy_helper.from_array(w2, name="W2"),
            numpy_helper.from_array(b2, name="b2"),
        ],
    )
    model = helper.make_model(
        graph,
        producer_name="tt-xla-onnx-ws4-gemm-gap",
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
        default=Path(__file__).resolve().parent / "fixtures" / "mlp2_gemm.onnx",
        help="Output .onnx path",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(build_mlp_gemm_model(seed=args.seed), args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
