# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Layer-0 fused ``input_layernorm`` + ``self_attn`` PCC repro — no Janus/MMGPT load at test time.

**Canonical sanity (real decode I/O + HF weights):**

1. Capture once (CPU, no TT)::

     pytest -s tests/torch/models/janus_pro_pcc_drop_no_dep/test_save_layer0_no_dep_fixtures.py::test_save_layer0_no_dep_fixtures_pro_1b

2. Run TT op test (saved embeds + KV + weights + config; **no** ``tt_forge_models``)::

     pytest -m "n150 and single_device" -s \\
       tests/torch/models/janus_pro_pcc_drop_no_dep/test_layer0_ln_attn_no_dep.py::test_layer0_ln_attn_no_dep_pro_1b

3. Gate codegen graph (isolated Forge vs CPU, same as step 2; expect ~0.99 on ``self_attn``)::

     python examples/pytorch/codegen/python/janus_layer0_ln_attn_no_dep_compare.py

4. Export TTNN for tt-metal (after step 3 passes)::

     python examples/pytorch/codegen/python/janus_layer0_ln_attn_no_dep.py

Fixtures: ``janus_logs/layer0_tensors/<variant>/`` (``torch`` + ``transformers`` only at steps 2–4).

**Random baseline:** ``test_layer0_ln_attn_no_dep_pro_1b_random``.
"""

from __future__ import annotations

import os

import pytest

from tests.torch.models.janus_pro_pcc_drop_no_dep.arch_specs import get_layer0_spec
from tests.torch.models.janus_pro_pcc_drop_no_dep.op_test import (
    run_layer0_ln_attn_forge_vs_cpu_isolated,
)
from tests.torch.models.janus_pro_pcc_drop_no_dep.saved_fixtures import (
    fixture_dir_for_variant,
    saved_fixtures_available,
)
from tests.torch.models.janus_pro_pcc_drop_no_dep.tt_device_warmup import (
    ensure_tt_device_ready,
)


def _run_ln_attn_no_dep(
    variant: str,
    *,
    use_saved_inputs: bool,
    load_hf_weights: bool,
) -> None:
    ensure_tt_device_ready()

    spec = get_layer0_spec(variant)
    for line in spec.summary_lines():
        print(line)
    if use_saved_inputs:
        print(f"fixtures={fixture_dir_for_variant(variant)}")

    run_layer0_ln_attn_forge_vs_cpu_isolated(
        f"layer0_ln_attn_no_dep_{variant}",
        spec,
        use_saved_inputs=use_saved_inputs,
        load_hf_weights=load_hf_weights,
        assert_on_failure=False,
    )


def _require_saved_fixtures(variant: str) -> None:
    if saved_fixtures_available(variant):
        return
    pytest.skip(
        f"Saved decode fixtures missing for {variant} under {fixture_dir_for_variant(variant)}. "
        f"Run: pytest -s tests/torch/models/janus_pro_pcc_drop_no_dep/"
        f"test_save_layer0_no_dep_fixtures.py::test_save_layer0_no_dep_fixtures_{variant.lower()}"
    )


@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.n150
@pytest.mark.p150
def test_layer0_ln_attn_no_dep_pro_1b():
    """Pro-1B: saved decode I/O + weights; isolated Forge vs CPU (expect ~0.99 on ``self_attn``)."""
    _require_saved_fixtures("Pro_1B")
    _run_ln_attn_no_dep("Pro_1B", use_saved_inputs=True, load_hf_weights=False)


@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.p150
def test_layer0_ln_attn_no_dep_pro_7b():
    """Pro-7B: saved decode I/O + weights."""
    _require_saved_fixtures("Pro_7B")
    _run_ln_attn_no_dep("Pro_7B", use_saved_inputs=True, load_hf_weights=False)


@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.n150
@pytest.mark.p150
def test_layer0_ln_attn_no_dep_pro_1b_random():
    """Pro-1B random weights/inputs (smoke only)."""
    use_hf = os.environ.get("JANUS_NO_DEP_HF_WEIGHTS", "0") == "1"
    _run_ln_attn_no_dep("Pro_1B", use_saved_inputs=False, load_hf_weights=use_hf)
