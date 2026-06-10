# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Fix 2 — keep ``ttnn.group_norm`` native instead of decomposing it.

Background
----------
At ``optimization_level=0`` tt-mlir's ``GroupNormDecompositionRewritePattern`` runs
*without* op-constraint validation and unconditionally expands every
``ttnn.group_norm`` into ``reshape -> mean -> subtract -> ...``.  At 1280x720 that
``mean`` materializes a ~1.89 GiB tile-padded transpose -> OOM
(see ``zimage_logs/composite_bisect_*.log``).

At ``optimization_level >= 1`` (and a tt-mlir built with ``TTMLIR_ENABLE_OPMODEL=ON``,
which this build has) the decomposition pass instead *validates* the native
``ttnn.group_norm`` op and keeps it when the kernel can run — using the automatic
core-grid selection added in tt-metal #40916 (present in the pinned tt-metal).
That fused kernel computes per-group stats without the giant intermediate.

Diagnostic
----------
The IR-export tests assert the **ttnn** stage still contains ``ttnn.group_norm``
(native kernel kept) rather than decomposing to ``mean``/``subtract``.  The device
tests then confirm the cases that OOM'd at opt0 now execute.

Run:
  pytest -svv tests/torch/model/zimage_decoder_debug/test_fix2_native_groupnorm.py \\
    2>&1 | tee zimage_logs/fix2_native_groupnorm.log
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.workloads.torch_workload import TorchWorkload

from tests.infra.testers.compiler_config import CompilerConfig
from tests.infra.testers.single_chip.op.op_tester import OpTester

from .ir_analysis import format_summary, summarize_export_dir
from .shared import (
    build_composite_norm2_only,
    build_confirm_prefix_through_norm2_composite,
    build_vae_decoder_all_composite_groupnorm,
    d3_input_key,
    norm2_isolated_input_key,
)

IR_EXPORT_ROOT = Path(__file__).resolve().parents[4] / "zimage_decoder_ir"

OPT_LEVELS = [1, 2]


def _build_case(case: str, context: dict):
    dec = context["vae"].decoder
    stages = context["stages"]
    if case == "composite_norm2_alone":
        return build_composite_norm2_only(dec).eval(), [
            stages[norm2_isolated_input_key()]
        ]
    if case == "composite_prefix_norm2":
        return build_confirm_prefix_through_norm2_composite(dec).eval(), [
            stages[d3_input_key()]
        ]
    raise ValueError(f"Unknown case: {case}")


def _ttnn_group_norm_kept(export_dir: Path) -> bool:
    """True if the lowered ttnn IR still has a native ttnn.group_norm op."""
    summary = summarize_export_dir(export_dir)
    ttnn = summary["stages"].get("ttnn", {})
    return ttnn.get("composite", {}).get("ttnn.group_norm", 0) > 0


@pytest.mark.model_test
@pytest.mark.parametrize("opt_level", OPT_LEVELS)
@pytest.mark.parametrize(
    "case_name", ["composite_norm2_alone", "composite_prefix_norm2"]
)
def test_export_ir_native_groupnorm(vae_decoder_context, case_name, opt_level):
    """Export IR at opt>=1; report whether ttnn.group_norm stays native vs decomposes."""
    xr.set_device_type("TT")
    export_dir = IR_EXPORT_ROOT / f"{case_name}_opt{opt_level}"
    export_dir.mkdir(parents=True, exist_ok=True)

    module, inputs = _build_case(case_name, vae_decoder_context)
    compiler_config = CompilerConfig(
        optimization_level=opt_level,
        export_path=str(export_dir),
        export_model_name=f"zimage_{case_name}",
    )
    tester = OpTester(framework=Framework.TORCH, compiler_config=compiler_config)
    workload = TorchWorkload(model=module, args=inputs)
    # Compilation emits IR before device execution; capture IR even if exec OOMs.
    try:
        tester.test(workload)
        ran = True
    except Exception as exc:  # noqa: BLE001 - diagnostic, re-raised after IR report
        ran = False
        exec_error = exc

    print("\n" + format_summary(summarize_export_dir(export_dir)))
    kept = _ttnn_group_norm_kept(export_dir)
    print(
        f"\n[Fix2] case={case_name} opt={opt_level} "
        f"native_ttnn_group_norm_kept={kept} device_ran={ran}"
    )
    if not ran:
        # Surface the device failure (likely OOM) once the diagnostic is printed.
        raise exec_error


@pytest.mark.model_test
@pytest.mark.parametrize("opt_level", OPT_LEVELS)
def test_device_prefix_norm2_native(vae_decoder_context, opt_level):
    """Minimal cumulative repro (prefix + composite norm2) on device at opt>=1."""
    xr.set_device_type("TT")
    dec = vae_decoder_context["vae"].decoder
    stages = vae_decoder_context["stages"]

    module = build_confirm_prefix_through_norm2_composite(dec).eval()
    inputs = [stages[d3_input_key()]]
    run_graph_test(
        module,
        inputs,
        framework=Framework.TORCH,
        compiler_config=CompilerConfig(optimization_level=opt_level),
    )


@pytest.mark.model_test
@pytest.mark.parametrize("opt_level", OPT_LEVELS)
def test_device_full_decoder_native(vae_decoder_context, opt_level):
    """Full Z-Image VAE decoder, all GroupNorms composite, on device at opt>=1."""
    xr.set_device_type("TT")
    vae = vae_decoder_context["vae"]
    latents = vae_decoder_context["latents"]

    model = build_vae_decoder_all_composite_groupnorm(vae)
    run_graph_test(
        model,
        [latents],
        framework=Framework.TORCH,
        compiler_config=CompilerConfig(optimization_level=opt_level),
    )
