# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Validate ``TTFusedMoE``'s DeepSeek-V4 routing against the ground-truth
reference ``Gate`` (the tt_forge_models DeepSeek-V4 ``modified_model``, which
implements the real ``sqrtsoftplus`` + hash / noaux_tc routing in pure torch).

vLLM's own DSV4 router is a CUDA-only kernel (``_moe_C.topk_softplus_sqrt``), so
this device-agnostic reference is how the TT reimplementation is checked — no
GPU required. Covers both hash-routed (layer < num_hash_layers) and score-routed
(noaux_tc bias) layers.
"""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("third_party.tt_forge_models.deepseek_v4.modified_model")
from vllm_tt.layers.fused_moe import TTFusedMoE  # noqa: E402

from third_party.tt_forge_models.deepseek_v4.modified_model import (  # noqa: E402
    model_decode_opt as mdo,
)

_DIM, _E, _TOPK, _VOCAB, _RS = 16, 8, 2, 12, 1.5


def _ref_gate(layer_id, n_hash, seed):
    args = mdo.ModelArgs(
        dim=_DIM,
        n_routed_experts=_E,
        n_activated_experts=_TOPK,
        score_func="sqrtsoftplus",
        route_scale=_RS,
        n_hash_layers=n_hash,
        vocab_size=_VOCAB,
    )
    g = torch.Generator().manual_seed(seed)
    with mdo.set_dtype(torch.float32):
        gate = mdo.Gate(layer_id, args).eval()
    with torch.no_grad():
        gate.weight.copy_(torch.randn(_E, _DIM, generator=g))
        if gate.hash:
            gate.tid2eid.copy_(
                torch.randint(0, _E, (_VOCAB, _TOPK), generator=g, dtype=torch.int32)
            )
        else:
            gate.bias.copy_(torch.randn(_E, generator=g))
    return gate


def _fake_ttmoe(gate):
    return SimpleNamespace(
        top_k=_TOPK,
        renormalize=True,
        scoring_func="sqrtsoftplus",
        e_score_correction_bias=None if gate.hash else gate.bias.data.clone(),
        hash_indices_table=gate.tid2eid.data.clone() if gate.hash else None,
        routed_scaling_factor=_RS,
    )


@pytest.mark.push
@pytest.mark.parametrize("hash_layer", [True, False], ids=["hash", "noaux_tc"])
def test_ttfusedmoe_routing_matches_reference_gate(hash_layer):
    layer_id, n_hash = (0, 1) if hash_layer else (1, 0)
    gate = _ref_gate(layer_id, n_hash, seed=0)
    assert gate.hash == hash_layer
    fake = _fake_ttmoe(gate)

    torch.manual_seed(1)
    T = 6
    x = torch.randn(T, _DIM)
    input_ids = torch.randint(0, _VOCAB, (T,))

    # Reference gate consumes hidden states; TTFusedMoE consumes gate logits.
    with torch.no_grad():
        ref_w, ref_idx = gate(x, input_ids)
    logits = F.linear(x.float(), gate.weight.float())
    w, idx = TTFusedMoE._route_sqrtsoftplus(fake, logits, input_ids)

    assert torch.equal(idx.long(), ref_idx.long()), "expert selection differs"
    assert torch.allclose(w.float(), ref_w.float(), atol=1e-5), "routing weights differ"
