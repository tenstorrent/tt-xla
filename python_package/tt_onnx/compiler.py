# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Compile StableHLO MLIR to a PJRT LoadedExecutable via the JAX tt plugin."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping

import jax
from jax._src import compiler as jax_compiler
from jax._src import xla_bridge as xla_bridge
from jaxlib import xla_client as xc

from .mlir_utils import canonicalize_onnx_stablehlo


@dataclass(frozen=True)
class CompileArtifacts:
    stablehlo_mlir: str
    compile_time_s: float
    compile_options: dict[str, str]
    export_path: str = ""
    export_model_name: str = ""


def get_tt_device() -> xc.Device:
    """
    Return the first TT JAX device.

    JAX loads the tt PJRT plugin via jax_plugins entry_points; do not call
    jax_plugin_tt.initialize() explicitly (that causes ALREADY_EXISTS).
    """
    devices = jax.devices("tt")
    if not devices:
        raise RuntimeError(
            "No TT JAX devices found. Ensure tt-xla is built, the reservation "
            "has a Tenstorrent device, and pjrt_plugin_tt.so matches libTTMLIRCompiler."
        )
    return devices[0]


def compile_stablehlo_mlir(
    mlir_text: str,
    compile_options: Mapping[str, str] | None = None,
    *,
    device: xc.Device | None = None,
) -> tuple[Any, CompileArtifacts]:
    """
    Compile StableHLO MLIR through tt-xla PJRT (same path as jax.jit compiler_options).

    Returns:
        (loaded_executable, compile_artifacts)
    """
    opts = dict(compile_options or {})
    # JAX serializes compile input to VHLO before PJRT; auto runs VHLO→StableHLO.
    opts.setdefault("mlir_input_format", "auto")

    prepared = canonicalize_onnx_stablehlo(mlir_text)
    tt_device = device or get_tt_device()
    backend = xla_bridge.get_backend("tt")

    xla_options = jax_compiler.get_compile_options(
        num_replicas=1,
        num_partitions=1,
        env_options_overrides=opts,
        backend=backend,
    )
    executable_devices = xc.DeviceList((tt_device,))

    start = time.perf_counter()
    loaded = backend.compile_and_load(
        prepared.encode("utf-8"),
        executable_devices,
        xla_options,
    )
    compile_time_s = time.perf_counter() - start

    artifacts = CompileArtifacts(
        stablehlo_mlir=prepared,
        compile_time_s=compile_time_s,
        compile_options=dict(opts),
        export_path=opts.get("export_path", ""),
        export_model_name=opts.get("export_model_name", ""),
    )
    return loaded, artifacts
