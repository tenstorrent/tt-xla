# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Per-layer decoder PCC with **isolated** CPU vs Forge (fair compare).

**Cumulative** — full decode stack; PCC after layer 1, 2, … N (error can accumulate).

**Standalone** — for each layer ``i``, CPU runs layers ``0..i-1`` to build inputs, then
only layer ``i`` is Forge-vs-CPU (~0.99 each if no layer-specific bug).

Pro-1B only by default (24 layers). Use ``pytest -s`` for tables.
"""

from __future__ import annotations

import pytest

from tests.torch.models.janus_pro_pcc_drop.decoder_layers_op_test import (
    load_decode_bundle_for_variant,
    run_decoder_cumulative_layer_profile_isolated,
    run_decoder_standalone_layer_profile_isolated,
)

PRO_1B_NUM_HIDDEN_LAYERS = 24

# Subset for quick smoke; full run uses ``test_decoder_standalone_all_layers_pro_1b``
STANDALONE_LAYER_CHECKPOINTS_PRO_1B = (0, 1, 5, 10, 15, 20, 23)


@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.n150
@pytest.mark.p150
def test_decoder_cumulative_layer_profile_isolated_pro_1b():
    """Cumulative PCC after each layer (isolated CPU vs Forge stacks)."""
    bundle = load_decode_bundle_for_variant("Pro_1B")
    run_decoder_cumulative_layer_profile_isolated(bundle, variant_name="Pro_1B")


@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.n150
@pytest.mark.p150
@pytest.mark.parametrize(
    "layer_idx",
    STANDALONE_LAYER_CHECKPOINTS_PRO_1B,
    ids=[f"layer_{i}" for i in STANDALONE_LAYER_CHECKPOINTS_PRO_1B],
)
def test_decoder_standalone_layer_isolated_pro_1b(layer_idx: int):
    """Single decoder layer ``i`` with CPU-captured inputs (isolated Forge vs CPU)."""
    bundle = load_decode_bundle_for_variant("Pro_1B")
    run_decoder_standalone_layer_profile_isolated(
        bundle,
        variant_name="Pro_1B",
        layer_indices=(layer_idx,),
    )


@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.n150
@pytest.mark.p150
def test_decoder_standalone_all_layers_pro_1b():
    """All 24 layers standalone (long; ~24 compiles)."""
    bundle = load_decode_bundle_for_variant("Pro_1B")
    run_decoder_standalone_layer_profile_isolated(
        bundle,
        variant_name="Pro_1B",
        layer_indices=tuple(range(PRO_1B_NUM_HIDDEN_LAYERS)),
    )
