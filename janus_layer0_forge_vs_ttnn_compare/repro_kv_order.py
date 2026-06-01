# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Reproduce why Experiment A (~0.77) disagrees with saved Forge artifacts (~0.99).

Hypothesis: op test runs CPU forward first, mutates ``past_key_values``, then Forge
runs on dirty KV. Fresh bundles / KV clone should show ~0.99 on ``self_attn``.

Usage (tt-xla venv, TT device)::

  export JANUS_LAYER0_FIXTURE_DIR=.../janus_logs/layer0_tensors/Pro_1B
  python janus_layer0_forge_vs_ttnn_compare/repro_kv_order.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_CODEGEN_PY = _REPO / "examples" / "pytorch" / "codegen" / "python"
if str(_CODEGEN_PY) not in sys.path:
    sys.path.insert(0, str(_CODEGEN_PY))

import torch
from infra.evaluators import ComparisonConfig
from infra.utilities import Framework
from infra.workloads.torch_workload import TorchWorkload
from janus_layer0_build import build_layer0_codegen_workload
from janus_layer0_forge_vs_ttnn_compare.metrics import print_comparison_table
from tests.infra.testers.single_chip.op.op_tester import OpTester
from tests.torch.models.janus_pro_pcc_drop.decoder_submodule_sanity import (
    clone_dynamic_cache,
)
from tests.torch.models.janus_pro_pcc_drop_no_dep.tt_device_warmup import (
    ensure_tt_device_ready,
)


def _workload_from_fresh(variant: str = "Pro_1B") -> TorchWorkload:
    w = build_layer0_codegen_workload(variant)
    return TorchWorkload(model=w.wrapper, args=[w.inputs])


def _run_cpu_then_forge_shared(variant: str = "Pro_1B") -> tuple[torch.Tensor, torch.Tensor]:
    """Same wrapper — Experiment A order (CPU mutates KV before Forge)."""
    w = build_layer0_codegen_workload(variant)
    wl = TorchWorkload(model=w.wrapper, args=[w.inputs])
    tester = OpTester(
        comparison_config=ComparisonConfig(assert_on_failure=False),
        framework=Framework.TORCH,
    )
    cpu_out = tester._device_runner.run_on_cpu(wl).detach().cpu()
    tester._compile_for_tt_device(wl)
    forge_out = tester._device_runner.run_on_tt_device(wl).detach().cpu()
    return cpu_out, forge_out


def _run_fresh_each(variant: str = "Pro_1B") -> tuple[torch.Tensor, torch.Tensor]:
    """New bundle per run — matches capture_forge / capture_cpu."""
    tester = OpTester(
        comparison_config=ComparisonConfig(assert_on_failure=False),
        framework=Framework.TORCH,
    )
    cpu_wl = _workload_from_fresh(variant)
    cpu_out = tester._device_runner.run_on_cpu(cpu_wl).detach().cpu()

    forge_wl = _workload_from_fresh(variant)
    tester._compile_for_tt_device(forge_wl)
    forge_out = tester._device_runner.run_on_tt_device(forge_wl).detach().cpu()
    return cpu_out, forge_out


def _run_cpu_then_forge_cloned_kv(variant: str = "Pro_1B") -> tuple[torch.Tensor, torch.Tensor]:
    """CPU then Forge on same wrapper, but reset KV before Forge."""
    w = build_layer0_codegen_workload(variant)
    wl = TorchWorkload(model=w.wrapper, args=[w.inputs])
    tester = OpTester(
        comparison_config=ComparisonConfig(assert_on_failure=False),
        framework=Framework.TORCH,
    )
    cpu_out = tester._device_runner.run_on_cpu(wl).detach().cpu()
    w.wrapper.past_key_values = clone_dynamic_cache(w.wrapper.past_key_values)
    tester._compile_for_tt_device(wl)
    forge_out = tester._device_runner.run_on_tt_device(wl).detach().cpu()
    return cpu_out, forge_out


def main(variant: str = "Pro_1B") -> None:
    ensure_tt_device_ready()

    scenarios = [
        ("A: CPU then Forge (shared KV)", _run_cpu_then_forge_shared),
        ("B: Fresh bundle per run", _run_fresh_each),
        ("C: CPU then Forge (KV cloned before Forge)", _run_cpu_then_forge_cloned_kv),
    ]

    for title, fn in scenarios:
        print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
        cpu_out, forge_out = fn(variant)
        rows = print_comparison_table(title, "CPU", "Forge", cpu_out, forge_out)
        attn = next(m for name, m in rows if name == "self_attn")
        print(f"  -> self_attn PCC = {attn.pcc:.4f}")

    print(
        "\nIf A ~0.77 and B ~0.99: old op_test shared-module bug (fixed in op_test.py isolated helper)."
    )
    print("If all ~0.99: fair Forge vs CPU; use isolated sanity for signoff.")


if __name__ == "__main__":
    main()
