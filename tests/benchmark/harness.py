# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Shared driver-layer harness for the torch-xla / TT perf benchmarks.

Every torch-xla benchmark driver (vision, encoder, imagegen, llm) repeats the
same boilerplate: select the Tenstorrent runtime, name the module export path,
assemble the custom compile-options dict, fill in the common device/backend
fields of the result, and assert PCC against the CPU golden. This module owns
that boilerplate so the drivers stay focused on their measurement logic.

``utils.py`` remains the lower-level layer (PCC math, naming, result schema);
this harness orchestrates those helpers for the driver layer.
"""

import socket

import torch_xla
import torch_xla.runtime as xr

from utils import compute_pcc, get_xla_device_arch

# Directory the compiler exports compiled TTNN modules to (consumed e.g. by the
# fusion checker). Shared by every torch-xla driver.
MODULE_EXPORT_PATH = "modules"

# Sentinel distinguishing "argument omitted" from an explicit ``None``: an
# omitted experimental option is left out of the dict entirely, while an
# explicit value (including None/""/False) is forwarded verbatim.
_UNSET = object()


def init_tt_runtime() -> None:
    """Select the Tenstorrent PJRT runtime. Idempotent; safe to call at import."""
    xr.set_device_type("TT")


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
    options = build_compile_options(**kwargs)
    torch_xla.set_custom_compile_options(options)
    return options


def tt_xla_device_fields() -> dict:
    """Common ``create_benchmark_result`` device/backend kwargs for TT drivers."""
    return {
        "program_cache_enabled": True,
        "torch_xla_enabled": True,
        "backend": "tt",
        "device_name": socket.gethostname(),
        "arch": get_xla_device_arch(),
        "device_count": xr.global_runtime_device_count(),
    }


def assert_pcc(device_output, golden_output, required_pcc: float) -> float:
    """Compute PCC against the CPU golden and assert it meets ``required_pcc``."""
    pcc_value = compute_pcc(device_output, golden_output)
    assert (
        pcc_value >= required_pcc
    ), f"PCC comparison failed. PCC={pcc_value:.6f}, Required={required_pcc}"
    print(f"PCC verification passed with PCC={pcc_value:.6f}")
    return pcc_value
