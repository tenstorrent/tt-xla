# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Execute PJRT LoadedExecutable and ONNX Runtime CPU reference."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np


def run_loaded_executable(
    loaded_executable: Any,
    device: jax.Device,
    inputs: Sequence[np.ndarray],
) -> list[np.ndarray]:
    """
    Run a PJRT LoadedExecutable with host numpy inputs on a TT device.

    Args:
        loaded_executable: xla_client.LoadedExecutable from compile_stablehlo_mlir.
        device: jax device (typically jax.devices('tt')[0]).
        inputs: Ordered input arrays matching the public MLIR function signature.
    """
    device_inputs = [
        jax.device_put(jnp.asarray(value), device) for value in inputs
    ]
    result = loaded_executable.execute_sharded(device_inputs, with_tokens=False)
    output_shards = result.disassemble_into_single_device_arrays()
    return [np.asarray(jax.device_get(shard[0])) for shard in output_shards]


def run_onnxruntime_cpu(
    onnx_path: str,
    feed: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Reference execution via ONNX Runtime (CPU)."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is required for reference checks. "
            "Install with: pip install onnxruntime"
        ) from exc

    session = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
    )
    from .feed_utils import prepare_feed_for_ort

    ort_inputs = prepare_feed_for_ort(session, feed)
    output_names = [out.name for out in session.get_outputs()]
    outputs = session.run(output_names, ort_inputs)
    return dict(zip(output_names, outputs))


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
