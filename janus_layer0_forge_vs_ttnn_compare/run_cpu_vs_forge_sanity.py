# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Experiment A: same check as ``test_layer0_ln_attn_no_dep_pro_1b`` (Forge vs CPU, no pytest).

Run from tt-xla root with TT device::

  python janus_layer0_forge_vs_ttnn_compare/run_cpu_vs_forge_sanity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from infra.evaluators import ComparisonConfig
from infra.utilities import Framework
from infra.workloads.torch_workload import TorchWorkload
from janus_layer0_forge_vs_ttnn_compare.metrics import STAGE_NAMES, print_comparison_table
from tests.infra.testers.single_chip.op.op_tester import OpTester
from tests.torch.models.janus_pro_pcc_drop.decoder_op_test_utils import (
    _split_stage_outputs,
    _tensor_match_metrics,
    TorchComparisonEvaluator,
)
from tests.torch.models.janus_pro_pcc_drop_no_dep.arch_specs import get_layer0_spec
from tests.torch.models.janus_pro_pcc_drop_no_dep.build_modules import (
    Layer0LnAttnNoDep,
    build_layer0_no_dep,
)
from tests.torch.models.janus_pro_pcc_drop_no_dep.saved_fixtures import saved_fixtures_available
from tests.torch.models.janus_pro_pcc_drop_no_dep.tt_device_warmup import ensure_tt_device_ready


def main(variant: str = "Pro_1B") -> None:
    if not saved_fixtures_available(variant):
        raise SystemExit(f"Missing fixtures for {variant}")

    ensure_tt_device_ready()
    spec = get_layer0_spec(variant)
    bundle = build_layer0_no_dep(spec, use_saved_inputs=True, load_hf_weights=False)
    wrapper = Layer0LnAttnNoDep(bundle)

    workload = TorchWorkload(model=wrapper, args=[bundle.inputs_embeds])
    tester = OpTester(
        comparison_config=ComparisonConfig(assert_on_failure=False),
        framework=Framework.TORCH,
    )
    cpu_out = tester._device_runner.run_on_cpu(workload)
    tester._compile_for_tt_device(workload)
    forge_out = tester._device_runner.run_on_tt_device(workload)

    cpu_t = cpu_out.detach().cpu()
    forge_t = forge_out.detach().cpu()

    print_comparison_table(
        f"Experiment A — Forge vs CPU ({variant})",
        reference_label="CPU (eager)",
        actual_label="Forge (TT device)",
        reference=cpu_t,
        actual=forge_t,
    )

    evaluator = TorchComparisonEvaluator(ComparisonConfig(assert_on_failure=False))
    cpu_stages = _split_stage_outputs(cpu_t, len(STAGE_NAMES))
    forge_stages = _split_stage_outputs(forge_t, len(STAGE_NAMES))
    attn = _tensor_match_metrics(evaluator, forge_stages[2], cpu_stages[2])
    print(f"\nExperiment A summary: self_attn PCC = {attn.pcc:.4f} (expect ~0.77 for repro)")


if __name__ == "__main__":
    main()
