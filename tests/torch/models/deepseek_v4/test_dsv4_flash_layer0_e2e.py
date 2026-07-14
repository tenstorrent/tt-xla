# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""First-layer end-to-end test for DeepSeek-V4-Flash on Tenstorrent.

DeepSeek-V4-Flash layer 0 is **SWA-only** (``compress_ratios[0] == 0`` — no
compressed/indexer branch), so it exercises exactly the sliding-window MLA path.
This test:

1. Verifies from the real model config that layer 0 is SWA-only.
2. Loads layer 0's real attention weights (fp8-block linears dequantized to
   bf16 by ``weight_loader``; a single ~3.5 GB shard) and runs the
   ``modified_model`` layer-0 ``Attention`` on the **TT device vs CPU**,
   asserting they match (device-correctness on the real weight distribution).

This is the focused single-layer counterpart to ``test_deepseek_v4_e2e_streaming``
(which streams all 43 layers on a galaxy). The vLLM-engine path is not used here:
the DSV4-Flash checkpoint ships in DeepSeek *native* tensor naming (``layers.N.
attn.wq_a`` …), which vLLM's HF-format ``DeepseekV4ForCausalLM`` loader does not
consume, and layer 0 is additionally a hash-routed MoE layer; the torch
``modified_model`` is the path this checkpoint is built for.
"""
import os
import sys
from types import SimpleNamespace

import pytest
import torch

# modified_model + the sibling weight_loader (mirrors the streaming test).
pytest.importorskip("third_party.tt_forge_models.deepseek_v4.modified_model")
sys.path.insert(0, os.path.dirname(__file__))
import weight_loader as wl  # noqa: E402

from third_party.tt_forge_models.deepseek_v4.modified_model import (  # noqa: E402
    model_decode_opt as mdo,
)

MODEL_NAME = "deepseek-ai/DeepSeek-V4-Flash"
_SEQ_LEN = 32  # prefill, <= window_size (128)
_PCC_BAR = 0.98


def _load_args():
    try:
        args = wl.load_config_args(MODEL_NAME, force_bf16=True)
    except Exception as e:  # no network / gated repo
        pytest.skip(f"cannot load {MODEL_NAME} config: {e}")
    args.max_batch_size = 1
    args.max_seq_len = 64
    return args


def _build_layer0_attention(args):
    """Layer-0 Attention with real fp8->bf16 attention weights loaded."""
    with mdo.set_dtype(torch.bfloat16):
        attn = mdo.Attention(0, args).eval()
    # Attention-only subset (avoids dequanting the 256 MoE experts in the shard).
    try:
        raw = wl._load_raw_subset(MODEL_NAME, ["layers.0.attn."])
    except Exception as e:
        pytest.skip(f"cannot fetch {MODEL_NAME} layer-0 weights: {e}")
    state = wl._dequant_paired(raw, "layers.0.attn.")
    missing, _ = attn.load_state_dict(state, strict=False)
    own = set(attn.state_dict().keys())
    unloaded = [
        k for k in missing if k in own and not k.endswith(("kv_cache", "freqs_cis"))
    ]
    assert not unloaded, f"layer-0 attention params not loaded: {unloaded[:8]}"
    return attn


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    va, vb = a - a.mean(), b - b.mean()
    return float((va @ vb) / (va.norm() * vb.norm() + 1e-12))


@pytest.mark.nightly
def test_dsv4_flash_layer0_is_swa_only():
    """Layer 0 must be SWA-only (compress_ratio 0/1, no compressed branch)."""
    args = _load_args()
    ratios = list(args.compress_ratios)
    assert ratios[0] <= 1, (
        f"layer 0 is not SWA-only: compress_ratios[0]={ratios[0]} "
        f"(pattern={ratios[:8]})"
    )


@pytest.mark.nightly
@pytest.mark.model_test
def test_dsv4_flash_layer0_attention_on_device():
    """Run layer-0 SWA attention on the TT device with real weights; require it
    to match the CPU reference."""
    import torch_xla
    import torch_xla.runtime as xr

    xr.set_device_type("TT")
    if xr.global_runtime_device_count() < 1:
        pytest.skip("no TT device available")

    args = _load_args()
    assert args.compress_ratios[0] <= 1, "layer 0 must be SWA-only"

    torch.manual_seed(0)
    x = (torch.randn(1, _SEQ_LEN, args.dim) * 0.1).to(torch.bfloat16)

    # CPU reference.
    attn_cpu = _build_layer0_attention(args)
    with torch.no_grad():
        o_cpu = attn_cpu(x.clone(), torch.tensor(0)).float()
    assert torch.isfinite(o_cpu).all(), "CPU layer-0 output has NaN/Inf"

    # TT device (fresh instance; forward mutates the kv_cache buffer).
    dev = torch_xla.device()
    attn_dev = _build_layer0_attention(args).to(dev)
    with torch.no_grad():
        o_dev = attn_dev(x.clone().to(dev), torch.tensor(0, device=dev))
        torch_xla.sync()
        o_dev = o_dev.cpu().float()
    assert torch.isfinite(o_dev).all(), "TT layer-0 output has NaN/Inf"

    pcc = _pcc(o_cpu, o_dev)
    print(
        f"\nDSV4-Flash layer-0 SWA attention: out={tuple(o_dev.shape)} "
        f"tt-vs-cpu PCC={pcc:.5f}"
    )
    assert pcc > _PCC_BAR, f"layer-0 SWA attention tt-vs-cpu PCC {pcc:.5f} < {_PCC_BAR}"


@pytest.mark.nightly
@pytest.mark.model_test
def test_dsv4_flash_layer0_moe_hash_routing_runs():
    """Layer 0's real MoE runs with **hash routing**: 256 fp4 experts
    dequantized to bf16 + the ``tid2eid`` token→expert table.

    This is the reference (``modified_model``) MoE on CPU with the real layer-0
    ffn weights — the full 256-expert MoE (~13 GB bf16) exceeds a single
    Wormhole's DRAM, so the *on-device* MoE uses the multi-device-sharded
    sparse-MLP path that ``test_deepseek_v4_e2e_streaming`` already exercises
    (layer 0 included). The TT vLLM ``TTFusedMoE`` reproduction of this hash
    routing is checked in ``test_dsv4_moe_reference_parity`` and run on the TT
    device by ``test_dsv4_flash_layer0_hash_routing_on_device``.
    """
    args = _load_args()
    assert args.n_hash_layers >= 1, "layer 0 should be a hash-MoE layer"

    with mdo.set_dtype(torch.bfloat16):
        moe = mdo.MoE(0, args).eval()
    assert moe.gate.hash, "layer-0 gate must be hash-routed (tid2eid)"

    try:
        raw = wl._load_raw_subset(MODEL_NAME, ["layers.0.ffn."])
    except Exception as e:
        pytest.skip(f"cannot fetch {MODEL_NAME} layer-0 ffn weights: {e}")
    state = wl._dequant_paired(raw, "layers.0.ffn.")
    missing, unexpected = moe.load_state_dict(state, strict=False)
    assert not unexpected, f"unexpected MoE keys: {sorted(unexpected)[:8]}"
    # gate + tid2eid + shared + at least the routed experts must have loaded.
    assert not [k for k in missing if k.startswith(("gate.", "shared_experts."))]

    torch.manual_seed(0)
    tok = 4
    x = (torch.randn(tok, args.dim) * 0.1).to(torch.bfloat16)
    input_ids = torch.randint(0, args.vocab_size, (tok,))
    with torch.no_grad():
        y = moe(x, input_ids)
    assert y.shape == x.shape
    assert torch.isfinite(y.float()).all(), "layer-0 MoE output has NaN/Inf"

    # The gate is hash-routed: the selected experts are exactly tid2eid[token].
    with torch.no_grad():
        _, indices = moe.gate(x.float(), input_ids)
    assert torch.equal(indices.long(), moe.gate.tid2eid[input_ids].long())
    print(
        f"\nDSV4-Flash layer-0 MoE (hash): out={tuple(y.shape)} finite=True; "
        f"experts/token from tid2eid, e.g. token0 -> {indices[0].tolist()}"
    )


@pytest.mark.nightly
def test_dsv4_flash_layer0_hash_routing_on_device():
    """The TT vLLM hash-routing (``TTFusedMoE._route_sqrtsoftplus``) lowers and
    runs on the TT device, matching CPU."""
    import torch_xla
    import torch_xla.runtime as xr
    from vllm_tt.layers.fused_moe import TTFusedMoE

    xr.set_device_type("TT")
    if xr.global_runtime_device_count() < 1:
        pytest.skip("no TT device available")

    E, topk, vocab, tok = 8, 2, 5, 4
    torch.manual_seed(0)
    tid2eid = torch.randint(0, E, (vocab, topk), dtype=torch.int32)
    logits = torch.randn(tok, E)
    input_ids = torch.randint(0, vocab, (tok,))

    def _fake(table, dev=None):
        return SimpleNamespace(
            top_k=topk,
            renormalize=True,
            scoring_func="sqrtsoftplus",
            e_score_correction_bias=None,
            hash_indices_table=table if dev is None else table.to(dev),
            routed_scaling_factor=1.5,
        )

    w_cpu, idx_cpu = TTFusedMoE._route_sqrtsoftplus(_fake(tid2eid), logits, input_ids)

    dev = torch_xla.device()
    w_dev, idx_dev = TTFusedMoE._route_sqrtsoftplus(
        _fake(tid2eid, dev), logits.to(dev), input_ids.to(dev)
    )
    torch_xla.sync()
    idx_dev, w_dev = idx_dev.cpu(), w_dev.cpu().float()

    assert torch.equal(idx_dev.long(), idx_cpu.long()), "device hash indices differ"
    assert torch.allclose(w_dev, w_cpu.float(), atol=1e-3), "device hash weights differ"
    print(f"\nhash routing on TT: indices match, weights PCC-close (tok={tok})")
