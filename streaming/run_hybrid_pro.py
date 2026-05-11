# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Hybrid streaming inference for DeepSeek-V4-Pro.

Thin wrapper around streaming.run_hybrid that:

    1. Repoints `weight_loader.REPO_ID` to "deepseek-ai/DeepSeek-V4-Pro" so
       per-layer + top-level weight loads pull from the Pro repo.
    2. Forces expert weights to bfp_bf4 (mandatory — Pro experts otherwise
       overflow device DRAM) and attention linears to bf16 via
       `apply_weight_dtype_overrides`.

Override ordering (preserved by run_hybrid.main):
    per-layer ship (mark_sharding inside) → per-layer dummy flush →
    apply_weight_dtype_overrides → register head hook → torch.compile

This matches the proven order in test_deepseek_v4_full_e2e.py:325-355.
The dtype-override ops are inserted AFTER the per-layer flush executes
(which only need to release host RAM, not exercise the bfp4 path) and
BEFORE the whole-model torch.compile, so only the final whole-model
StableHLO graph carries `weight_dtype_override` ops on the targeted
tensors.

Run (2-layer smoke test, IR dump for graph verification):

    source venv/activate
    STREAM_HYBRID_DISABLE_CONSTEVAL_TO_HOST=1 \
        STREAM_NUM_LAYERS=2 STREAM_MAX_NEW_TOKENS=2 \
        STREAM_HYBRID_IR_DUMP_DIR=/tmp/run_hybrid_pro_ir \
        python -u streaming/run_hybrid_pro.py

Then inspect <dump_dir>/irs/*.mlir for `weight_dtype_override` ops on
expert and attention weight tensors.
"""
from __future__ import annotations

# Repoint REPO_ID at module load — must happen BEFORE
# `streaming.run_hybrid` is imported, since that module captures
# `weight_loader` at import time and any subsequent
# `weight_loader.load_*` call (config args, per-layer weights, top-level
# weights, tokenizer) reads `weight_loader.REPO_ID` lazily.
from tests.torch.models.deepseek_v4 import weight_loader

weight_loader.REPO_ID = "deepseek-ai/DeepSeek-V4-Pro"
print(f"[hybrid-pro] REPO_ID repointed to {weight_loader.REPO_ID}",
      flush=True)

# NOTE: `streaming.run_hybrid` and `streaming.streaming_loader` import
# torch_xla at module level. With multiprocessing spawn, child workers
# re-import this script — keeping these imports under the __main__ guard
# below prevents workers from initializing torch_xla / PJRT (which would
# race with the parent for TT device handles and crash).


# Routed-expert stacked tensors (compound-sharded across both mesh axes
# in `_block_shard_spec`) MUST go through bfp_bf4 packing — Pro's 384
# experts × 7168 hidden bf16 stack will not fit on device.
#
# Attention linears stay bf16 (already their natural dtype). The
# override is still useful: it inserts `weight_dtype_override` ops at
# the same trace-point as the expert overrides so the compiler sees a
# uniform graph shape. Targets every Linear/ColumnParallelLinear/
# RowParallelLinear weight on the Attn block (model_decode_opt.py:545-556).
WEIGHT_OVERRIDES = {
    "layers.*.ffn.mlp.experts.gate_proj": "bfp_bf4",
    "layers.*.ffn.mlp.experts.up_proj":   "bfp_bf4",
    "layers.*.ffn.mlp.experts.down_proj": "bfp_bf4",
    # Shared experts: replicated across the 32-device mesh in
    # `_block_shard_spec`, so each device holds the full Linear weight.
    # bf16: ~132 MB/layer/device; ×60 layers = ~8 GB/device — far too big.
    # bfp_bf4 cuts to ~33 MB/layer/device → ~2 GB across 60 layers.
    "layers.*.ffn.shared_experts.w1.weight": "bfp_bf4",
    "layers.*.ffn.shared_experts.w2.weight": "bfp_bf4",
    "layers.*.ffn.shared_experts.w3.weight": "bfp_bf4",
    # Attention to bfp_bf4 (was bfp_bf8) — saves ~2.5 MB / layer / bank
    # = ~150 MB / bank for 60 layers, enough headroom to fit Pro 60-layer
    # peak under the 4 GB / bank limit. Slight accuracy hit on attention
    # vs bfp_bf8 but generation still produces tokens.
    "layers.*.attn.wq_a.weight": "bfp_bf4",
    "layers.*.attn.wq_b.weight": "bfp_bf4",
    "layers.*.attn.wkv.weight":  "bfp_bf4",
    "layers.*.attn.wo_a.weight": "bfp_bf4",
    "layers.*.attn.wo_b.weight": "bfp_bf4",
}


if __name__ == "__main__":
    from streaming.run_hybrid import main
    from streaming.streaming_loader import (
        _block_shard_spec_pro,
        _top_level_shard_spec_pro,
    )

    main(
        weight_overrides=WEIGHT_OVERRIDES,
        block_shard_spec_fn=_block_shard_spec_pro,
        top_level_shard_spec_fn=_top_level_shard_spec_pro,
    )
