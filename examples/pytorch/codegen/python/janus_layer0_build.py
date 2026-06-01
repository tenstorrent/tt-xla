# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Build ``Layer0LnAttnNoDep`` from saved fixtures — **no** ``torch_xla`` / ``tt_torch``.

Safe to import from tt-metal ``cpu_reference`` (torch + transformers only).
``janus_layer0_codegen.py`` adds xla export on top of this module.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[4]

DEFAULT_VARIANT = "Pro_1B"

STAGE_NAMES: tuple[str, ...] = ("decode_setup", "input_layernorm", "self_attn")


def ensure_repo_on_path() -> Path:
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    return _REPO_ROOT


@dataclass(frozen=True)
class Layer0CodegenWorkload:
    variant: str
    wrapper: Any
    inputs: torch.Tensor
    fixture_dir: Path


def build_layer0_codegen_workload(variant: str = DEFAULT_VARIANT) -> Layer0CodegenWorkload:
    ensure_repo_on_path()

    from tests.torch.models.janus_pro_pcc_drop_no_dep.arch_specs import get_layer0_spec
    from tests.torch.models.janus_pro_pcc_drop_no_dep.build_modules import (
        Layer0LnAttnNoDep,
        build_layer0_no_dep,
    )
    from tests.torch.models.janus_pro_pcc_drop_no_dep.saved_fixtures import (
        fixture_dir_for_variant,
        saved_fixtures_available,
    )

    if not saved_fixtures_available(variant):
        raise FileNotFoundError(
            f"Missing fixtures under {fixture_dir_for_variant(variant)}. Run:\n"
            "  pytest -s tests/torch/models/janus_pro_pcc_drop_no_dep/"
            f"test_save_layer0_no_dep_fixtures.py::test_save_layer0_no_dep_fixtures_{variant.lower()}"
        )

    spec = get_layer0_spec(variant)
    bundle = build_layer0_no_dep(
        spec,
        use_saved_inputs=True,
        load_hf_weights=False,
    )
    wrapper = Layer0LnAttnNoDep(bundle)
    return Layer0CodegenWorkload(
        variant=variant,
        wrapper=wrapper,
        inputs=bundle.inputs_embeds,
        fixture_dir=fixture_dir_for_variant(variant),
    )


def run_forward_stacked(variant: str = DEFAULT_VARIANT) -> torch.Tensor:
    """CPU eager stacked stages — same tensor as no-dep sanity / compare gate CPU side."""
    workload = build_layer0_codegen_workload(variant)
    with torch.inference_mode():
        return workload.wrapper(workload.inputs)
