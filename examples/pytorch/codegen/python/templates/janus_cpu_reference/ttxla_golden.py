# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CPU stacked stages via the **same** ``Layer0LnAttnNoDep`` module used for codegen.

This is the CPU side of tt-xla ``run_op_test`` / ``test_layer0_ln_attn_no_dep_pro_1b``
(Experiment A reference), not a reimplemented copy.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

_CODEGEN_ROOT = Path(__file__).resolve().parents[1]


def resolve_ttxla_root() -> Path:
    env = os.environ.get("JANUS_TTXLA_ROOT")
    if env:
        return Path(env)
    candidates = [
        _CODEGEN_ROOT.parent.parent.parent / "31_may_yyz" / "tt-xla",
        Path("/proj_sw/user_dev/ctr-akannan/31_may_yyz/tt-xla"),
    ]
    for path in candidates:
        if (path / "tests" / "torch" / "models" / "janus_pro_pcc_drop_no_dep").is_dir():
            return path
    raise FileNotFoundError(
        "tt-xla repo not found for CPU golden. Set JANUS_TTXLA_ROOT to your tt-xla checkout."
    )


def run_layer0_ln_attn_no_dep_stacked(variant: str = "Pro_1B") -> torch.Tensor:
    """``Layer0LnAttnNoDep(bundle)(inputs_embeds)`` — identical to codegen / op_test CPU."""
    repo = resolve_ttxla_root()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from tests.torch.models.janus_pro_pcc_drop_no_dep.arch_specs import get_layer0_spec
    from tests.torch.models.janus_pro_pcc_drop_no_dep.build_modules import (
        Layer0LnAttnNoDep,
        build_layer0_no_dep,
    )
    from tests.torch.models.janus_pro_pcc_drop_no_dep.saved_fixtures import (
        saved_fixtures_available,
    )

    if not saved_fixtures_available(variant):
        raise FileNotFoundError(
            f"Missing fixtures for {variant}. Run test_save_layer0_no_dep_fixtures in tt-xla."
        )

    spec = get_layer0_spec(variant)
    bundle = build_layer0_no_dep(
        spec,
        use_saved_inputs=True,
        load_hf_weights=False,
    )
    model = Layer0LnAttnNoDep(bundle)
    with torch.inference_mode():
        return model(bundle.inputs_embeds)
