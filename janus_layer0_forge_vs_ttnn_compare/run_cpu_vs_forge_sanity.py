# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Experiment A: same check as ``test_layer0_ln_attn_no_dep_pro_1b`` (isolated Forge vs CPU).

Run from tt-xla root with TT device::

  python janus_layer0_forge_vs_ttnn_compare/run_cpu_vs_forge_sanity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tests.torch.models.janus_pro_pcc_drop_no_dep.arch_specs import get_layer0_spec
from tests.torch.models.janus_pro_pcc_drop_no_dep.op_test import (
    run_layer0_ln_attn_forge_vs_cpu_isolated,
)
from tests.torch.models.janus_pro_pcc_drop_no_dep.saved_fixtures import saved_fixtures_available
from tests.torch.models.janus_pro_pcc_drop_no_dep.tt_device_warmup import ensure_tt_device_ready


def main(variant: str = "Pro_1B") -> None:
    if not saved_fixtures_available(variant):
        raise SystemExit(f"Missing fixtures for {variant}")

    ensure_tt_device_ready()
    spec = get_layer0_spec(variant)
    for line in spec.summary_lines():
        print(line)

    rows = run_layer0_ln_attn_forge_vs_cpu_isolated(
        f"Experiment A — Forge vs CPU ({variant})",
        spec,
        use_saved_inputs=True,
        load_hf_weights=False,
        return_metrics=True,
    )
    if rows is None:
        raise RuntimeError("no metrics returned")
    attn = dict(rows)["self_attn"]
    print(
        f"\nExperiment A summary: self_attn PCC = {attn.pcc:.4f} "
        "(isolated bundles; expect ~0.99, not false ~0.77 from shared KV)"
    )


if __name__ == "__main__":
    main()
