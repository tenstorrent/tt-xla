# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the MRv2 ``TTModelState.prepare_attn`` assembler.

``prepare_attn`` (see vllm_tt/model_state.py) packages the per-step device
arrays the runner computes host-side into a single ``TTMetadata`` and fans it
out to every attention layer -- mirroring the v1 fork's
``dict.fromkeys(self._attention_layer_names, attn_metadata)``. It uses no
instance state, so these run on cpu with no TT hardware and no model
(the state object is allocated without ``__init__``).
"""

import pytest
import torch

from vllm_tt.attention_impls.attention import TTMetadata
from vllm_tt.model_state import TTModelState


def _bare_state():
    # prepare_attn reads no self.* state; skip the heavy VllmConfig __init__.
    return object.__new__(TTModelState)


@pytest.mark.push
@pytest.mark.cpu
def test_prepare_attn_fans_out_shared_metadata_per_layer():
    ms = _bare_state()
    layers = ["layer.0", "layer.1", "layer.2"]
    page_table = torch.zeros(4, 8, dtype=torch.int32)
    cache_position = torch.zeros(4, dtype=torch.int32)
    fill_page_table = torch.ones(4, 8, dtype=torch.int32)
    batch_idx = torch.arange(4, dtype=torch.int32)

    out = ms.prepare_attn(
        layers,
        page_table=page_table,
        cache_position=cache_position,
        fill_page_table=fill_page_table,
        batch_idx=batch_idx,
        num_users=4,
        dp_size=2,
    )

    assert set(out.keys()) == set(layers)
    metas = list(out.values())
    # One shared TTMetadata object across all layers (matches dict.fromkeys).
    assert all(m is metas[0] for m in metas)

    m = metas[0]
    assert isinstance(m, TTMetadata)
    assert m.num_users == 4
    assert m.dp_size == 2
    assert m.is_causal is True
    assert m.attn_mask is None
    assert m.chunk_start_idx is None
    assert m.page_table is page_table
    assert m.fill_page_table is fill_page_table
    assert m.cache_position is cache_position
    assert m.batch_idx is batch_idx


@pytest.mark.push
@pytest.mark.cpu
def test_prepare_attn_fill_page_table_defaults_to_page_table():
    ms = _bare_state()
    page_table = torch.zeros(2, 4, dtype=torch.int32)

    out = ms.prepare_attn(
        ["only.layer"],
        page_table=page_table,
        cache_position=torch.zeros(2, dtype=torch.int32),
        num_users=2,
    )

    m = out["only.layer"]
    # No prefix roll supplied -> fill_page_table mirrors page_table.
    assert m.fill_page_table is page_table
    assert m.dp_size == 1
    assert m.batch_idx is None
