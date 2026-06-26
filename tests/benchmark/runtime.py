# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Execution layer: selecting the Tenstorrent runtime, configuring the compiler,
and reading device properties.

This is the shared benchmark module that talks to the live runtime (it selects
the torch-xla runtime, registers custom compile options, and queries device
arch / count). Everything here has side effects on global state or depends on a
running device — keep pure, device-free helpers in the other modules
(``reporting``, ``accuracy``, ``naming``, ``model_utils``).

``torch_xla`` / ``jax`` are imported lazily inside the functions that need them
so that, e.g., the JAX resnet benchmark can pull ``get_jax_device_arch`` without
dragging the torch-xla PJRT client into a JAX process (two clients in one
process is exactly what we want to avoid).
"""

import socket

# Directory the compiler exports compiled TTNN modules to (consumed e.g. by the
# fusion checker). Shared by every benchmark.
MODULE_EXPORT_PATH = "modules"

# Sentinel distinguishing "argument omitted" from an explicit ``None``: an
# omitted experimental option is left out of the dict entirely, while an
# explicit value (including None/""/False) is forwarded verbatim.
_UNSET = object()


def init_tt_runtime() -> None:
    """Select the Tenstorrent PJRT runtime. Idempotent; safe to call at import."""
    import torch_xla.runtime as xr

    xr.set_device_type("TT")


def align_arch(arch: str) -> str:
    """Align architecture name to standard format."""
    for item in ["wormhole", "blackhole"]:
        if item in arch:
            return item
    return ""


def get_jax_device_arch() -> str:
    """Get the architecture of the first JAX TT device."""
    import jax

    devices = jax.devices("tt")
    for device in devices:
        arch_name = str(device.device_kind).lower()
        return align_arch(arch_name)

    return ""


def get_xla_device_arch() -> str:
    """Get the architecture of the XLA device."""
    import torch_xla.runtime as xr

    # Query the physical runtime devices directly. This works in both regular
    # and SPMD modes. xm.xla_device_kind() cannot be used because in SPMD mode
    # (e.g. tensor-parallel benchmarks) xm.xla_device() resolves to a virtual
    # "SPMD:0" device that the device-kind lookup cannot find.
    attrs = xr.global_runtime_device_attributes()
    if not attrs:
        return ""
    arch_name = str(attrs[0]["device_arch"]).lower()
    return align_arch(arch_name)


def build_compile_options(
    *,
    optimization_level,
    export_model_name,
    ttnn_perf_metrics_output_file,
    enable_trace,
    experimental_weight_dtype=_UNSET,
    experimental_enable_permute_matmul_fusion=_UNSET,
    fp32_dest_acc_en=None,
    experimental_kv_cache_dtype=None,
    enable_create_d2m_subgraphs=False,
) -> dict:
    """Assemble the torch-xla custom compile-options dict.

    The base keys are always present. The ``experimental_*`` weight-dtype and
    permute-fusion keys are only emitted when explicitly passed (drivers that
    don't expose them omit them entirely). The remaining optionals follow the
    historical "include only when meaningfully set" rule.
    """
    options = {
        "optimization_level": optimization_level,
        "enable_trace": enable_trace,
        "export_path": MODULE_EXPORT_PATH,
        "export_model_name": export_model_name,
        "ttnn_perf_metrics_enabled": True,
        "ttnn_perf_metrics_output_file": ttnn_perf_metrics_output_file,
    }
    if experimental_weight_dtype is not _UNSET:
        options["experimental_weight_dtype"] = experimental_weight_dtype
    if experimental_enable_permute_matmul_fusion is not _UNSET:
        options["experimental_enable_permute_matmul_fusion"] = (
            experimental_enable_permute_matmul_fusion
        )
    if fp32_dest_acc_en is not None:
        options["fp32_dest_acc_en"] = fp32_dest_acc_en
    if experimental_kv_cache_dtype is not None:
        options["experimental-kv-cache-dtype"] = experimental_kv_cache_dtype
    if enable_create_d2m_subgraphs:
        options["enable_create_d2m_subgraphs"] = enable_create_d2m_subgraphs
    return options


def set_compile_options(**kwargs) -> dict:
    """Build the compile-options dict and register it with torch-xla.

    Returns the dict so callers that need to forward it (e.g. diffusion
    pipelines that merge extra options) can do so.
    """
    import torch_xla

    options = build_compile_options(**kwargs)
    torch_xla.set_custom_compile_options(options)
    return options


def tt_xla_device_fields() -> dict:
    """Common ``create_benchmark_result`` device/backend kwargs for TT drivers."""
    import torch_xla.runtime as xr

    return {
        "program_cache_enabled": True,
        "torch_xla_enabled": True,
        "backend": "tt",
        "device_name": socket.gethostname(),
        "arch": get_xla_device_arch(),
        "device_count": xr.global_runtime_device_count(),
    }
