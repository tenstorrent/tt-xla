#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Generate a trivial single-Add ONNX model for onnx-mlir smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install with: pip install onnx numpy"
    ) from exc


def build_add_model() -> onnx.ModelProto:
    input_a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 4])
    input_b = helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, 4])
    output = helper.make_tensor_value_info("C", TensorProto.FLOAT, [1, 4])

    add_node = helper.make_node("Add", inputs=["A", "B"], outputs=["C"], name="add0")
    graph = helper.make_graph(
        [add_node],
        "add_graph",
        [input_a, input_b],
        [output],
    )
    model = helper.make_model(
        graph,
        producer_name="tt-xla-onnx-smoke",
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
        default=Path(__file__).resolve().parent / "fixtures" / "add.onnx",
        help="Output .onnx path",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = build_add_model()
    onnx.save(model, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
