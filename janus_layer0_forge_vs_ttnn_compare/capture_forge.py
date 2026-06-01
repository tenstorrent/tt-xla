# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Run ``Layer0LnAttnNoDep`` on Forge/XLA (TT device) and save stacked stages.

Usage (tt-xla venv, device available)::

  cd /proj_sw/user_dev/ctr-akannan/31_may_yyz/tt-xla
  python janus_layer0_forge_vs_ttnn_compare/capture_forge.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_COMPARE_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _COMPARE_ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from infra.evaluators import ComparisonConfig
from infra.utilities import Framework
from infra.workloads.torch_workload import TorchWorkload
from tests.infra.testers.single_chip.op.op_tester import OpTester

from janus_layer0_forge_vs_ttnn_compare.paths import artifacts_dir, stacked_artifact
from tests.torch.models.janus_pro_pcc_drop_no_dep.arch_specs import get_layer0_spec
from tests.torch.models.janus_pro_pcc_drop_no_dep.build_modules import (
    Layer0LnAttnNoDep,
    build_layer0_no_dep,
)
from tests.torch.models.janus_pro_pcc_drop_no_dep.saved_fixtures import (
    saved_fixtures_available,
)
from tests.torch.models.janus_pro_pcc_drop_no_dep.tt_device_warmup import (
    ensure_tt_device_ready,
)


def _to_cpu_tensor(value: torch.Tensor) -> torch.Tensor:
    if hasattr(value, "detach"):
        return value.detach().cpu()
    return torch.as_tensor(value).cpu()


def capture_forge_stacked(variant: str = "Pro_1B") -> torch.Tensor:
    if not saved_fixtures_available(variant):
        raise FileNotFoundError(
            f"Missing fixtures for {variant}. Run test_save_layer0_no_dep_fixtures first."
        )

    ensure_tt_device_ready()
    spec = get_layer0_spec(variant)
    bundle = build_layer0_no_dep(
        spec,
        use_saved_inputs=True,
        load_hf_weights=False,
    )
    wrapper = Layer0LnAttnNoDep(bundle)
    inputs = [bundle.inputs_embeds]

    workload = TorchWorkload(model=wrapper, args=inputs)
    tester = OpTester(
        comparison_config=ComparisonConfig(assert_on_failure=False),
        framework=Framework.TORCH,
    )
    tester._compile_for_tt_device(workload)
    tt_out = tester._device_runner.run_on_tt_device(workload)
    stacked = _to_cpu_tensor(tt_out).to(torch.bfloat16)
    if stacked.ndim != 4 or stacked.shape[0] != 3:
        raise ValueError(f"Expected [3, B, S, H] from Forge, got {tuple(stacked.shape)}")
    return stacked


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture Forge/XLA stacked layer-0 stages.")
    parser.add_argument("--variant", default="Pro_1B", choices=["Pro_1B", "Pro_7B"])
    args = parser.parse_args()

    out_path = stacked_artifact("forge", args.variant)
    artifacts_dir().mkdir(parents=True, exist_ok=True)

    print(f"Running Forge on TT device ({args.variant}) ...")
    stacked = capture_forge_stacked(args.variant)
    torch.save(stacked, out_path)
    print(f"Wrote {out_path}  shape={tuple(stacked.shape)} dtype={stacked.dtype}")


if __name__ == "__main__":
    main()
