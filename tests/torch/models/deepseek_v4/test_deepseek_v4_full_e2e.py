# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end prefill+decode generation for DeepSeek-V4-Flash on TT.

Loads the full real published weights (43 layers, 256 routed experts,
n_mtp_layers=0) into `model_decode_opt.Transformer`, swaps each block's
MoE for an A2aSparseMLP via `tt_torch.sparse_mlp.enable_sparse_mlp`,
runs one prefill on the tokenized prompt and a fixed number of decode
steps, and prints the detokenized continuation.

No PCC; no CPU oracle. The only contract is that the loop produces tokens.

Compile budget: 4 tt-backend compiles total (2 const-eval + 2 main forward,
one prefill + one decode), via the same sp_buffer / one_buffer trick as
test_deepseek_v4_prefill_decode_loop_no_int.py:151-158.

Memory: assumes the host has enough RAM to hold the full dequantized
checkpoint (~600 GB BF16) plus per-block stacked expert copies during
sparse-MLP swap, before `.to(torch_xla.device())` ships shards out.
"""

from __future__ import annotations

import gc
from typing import Dict, Tuple

import numpy as np
import pytest
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra.utilities.torch_multichip_utils import enable_spmd
from torch_xla.distributed.spmd import Mesh
from tt_torch.sparse_mlp import enable_sparse_mlp
from tt_torch.weight_dtype import apply_weight_dtype_overrides

from tt_torch.sharding import sharding_constraint_hook

from third_party.tt_forge_models.deepseek_v4.modified_model import (
    model_decode_opt as mdo,
)

from . import weight_loader

# Prefill length must be a multiple of args.window_size (=128) so the
# attention's circular kv_cache update lands on a window boundary; otherwise
# the cache contents become inconsistent for the first decode step. It also
# must be >= max(args.compress_ratios) so the Compressor's
# `rope_idx = start_pos + 1 - ratio` stays non-negative
# (model_decode_opt.py:414). 128 satisfies both.
PROMPT_LEN = 128
MAX_NEW_TOKENS = 64

# Batch size is a hard requirement of the A2aSparseMLP op; on the (4, 8)
# mesh this shards to 8 rows per _axis_0 device.
BATCH_SIZE = 32

# Number of transformer blocks to load + run. Real config has 43; truncating
# to a prefix is a memory dial. Layer subset is sliced from
# real_args.compress_ratios so the included layers stay consistent with the
# real-checkpoint layout (mirrors transformer_args in
# test_deepseek_v4_tp_no_int.py:163).
NUM_LAYERS = 43

# see issue https://github.com/tenstorrent/tt-xla/issues/4444
torch._dynamo.config.cache_size_limit = 100

# One distinct prompt per batch row; len(PROMPTS) must equal BATCH_SIZE.
PROMPTS = [
    "How are you today?",
    "What is the capital of France?",
    "Explain machine learning briefly.",
    "Who painted the Mona Lisa?",
    "What is two plus two?",
    "Tell me a fun fact about space.",
    "What is photosynthesis?",
    "How does a transformer model work?",
    "What is the speed of light?",
    "Name three programming languages.",
    "What causes earthquakes?",
    "How do you make pasta from scratch?",
    "What is the largest planet in our solar system?",
    "Who wrote the play Hamlet?",
    "What is gravity?",
    "How does the internet work?",
    "What is the human brain made of?",
    "Tell me about black holes.",
    "What is DNA?",
    "How does a car engine work?",
    "What is the meaning of recursion?",
    "Who was Albert Einstein?",
    "What is climate change?",
    "How do plants grow?",
    "What is quantum entanglement?",
    "Tell me a short story about a robot.",
    "What is a relational database?",
    "How does Wi-Fi work?",
    "What is the Pythagorean theorem?",
    "Name three renewable energy sources.",
    "What was the French Revolution about?",
    "How do volcanoes form?",
]
assert len(PROMPTS) == BATCH_SIZE, (
    f"PROMPTS has {len(PROMPTS)} entries, must equal BATCH_SIZE={BATCH_SIZE}"
)


# ----------------------------------------------------------------------------
# All-at-once weight loader. Builds the Transformer skeleton, loads every
# real-checkpoint param in a single pass, then swaps every block's MoE for
# A2aSparseMLP.
#
# No gate / ffn patches needed: model_decode_opt.Gate.forward already has
# the _ambient_input_ids fallback and the SPMD-safe one_hot * scores form
# (model_decode_opt.py:678-714), and Block.forward already calls
# self.ffn(x, input_ids) (model_decode_opt.py:839), which the post-swap
# A2aSparseMLPWithSharedExperts accepts via its
# forward(hidden_states, *extra_args, **extra_kwargs) signature
# (sparse_mlp.py:993). RouterAdapter (sparse_mlp.py:753) flattens the 2D
# input_ids to 1D and threads them straight into Gate.forward, so the gate
# stash is also redundant.
# ----------------------------------------------------------------------------


def _build_and_load_full_model(args: mdo.ModelArgs, mesh_shape: Tuple[int, int]):
    """Construct Transformer, load all real weights in one call, then swap
    every block's MoE in one enable_sparse_mlp call. Returns the CPU-side
    model; caller is expected to .to(torch_xla.device()).
    """
    print(f"[build] constructing Transformer skeleton (n_layers={args.n_layers}, "
          f"n_routed_experts={args.n_routed_experts}) ...", flush=True)
    # Transformer relies on torch.get_default_dtype() for ParallelEmbedding
    # and the bf16 Linear weights; without this override they default to fp32
    # and mismatch the bf16 activations / checkpoint. Same construction
    # pattern as make_real_transformer in test_deepseek_v4_tp_no_int.py.
    prev_default = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = mdo.Transformer(args).eval()
    finally:
        torch.set_default_dtype(prev_default)

    print(f"[load] full state_dict for all {args.n_layers} layers ...", flush=True)
    sd = weight_loader.load_transformer_state_dict(
        range(args.n_layers), include_mtp=False
    )
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    del sd
    gc.collect()

    print(f"[swap] enable_sparse_mlp on all {args.n_layers} blocks ...", flush=True)
    enable_sparse_mlp(
        model,
        mesh=mesh_shape,
        cluster_axis=0,
        config=args,
        verbose=False,
    )

    print("[build] done. CPU model assembled.", flush=True)
    return model


# ----------------------------------------------------------------------------
# SPMD shard spec for the full model post-swap. Mesh axes are
# ("_axis_0", "_axis_1") to stay compatible with A2aSparseMLP's expert
# dispatch convention; attention specs map model->axis_1, batch->axis_0.
# ----------------------------------------------------------------------------


def make_mesh() -> Tuple[Mesh, Tuple[int, int]]:
    # Hardcoded 4x8 mesh; assumes a 32-device llmbox host.
    mesh_shape = (4, 8)
    num_devices = mesh_shape[0] * mesh_shape[1]
    assert xr.global_runtime_device_count() == num_devices, (
        f"Expected {num_devices} TT devices for mesh {mesh_shape}, got "
        f"{xr.global_runtime_device_count()}"
    )
    device_ids = np.array(range(num_devices))
    return Mesh(device_ids, mesh_shape, ("_axis_0", "_axis_1")), mesh_shape


def transformer_shard_spec(model: mdo.Transformer) -> Dict[torch.Tensor, Tuple]:
    """SPMD shard spec for every parameter and cache buffer in the post-swap
    Transformer. Mirrors `_attn_shard_spec` from
    test_deepseek_v4_tp_no_int.py:296 (with axis renames) for attention, and
    `get_shard_spec` from test_deepseek_v4_flash_moe.py:59 for MoE.
    """
    specs: Dict[torch.Tensor, Tuple] = {}

    # Top-level: replicate. Embed [vocab, dim] is large but sharding it would
    # need careful integration with ParallelEmbedding (out of scope here).
    specs[model.embed.weight] = (None, None)
    specs[model.norm.weight] = (None,)
    specs[model.head.weight] = (None, None)
    specs[model.hc_head_fn] = (None, None)
    specs[model.hc_head_base] = (None,)
    specs[model.hc_head_scale] = (None,)

    compound = ("_axis_0", "_axis_1")  # for compound-sharded experts

    for block in model.layers:
        # ---- Attention ----
        attn = block.attn
        specs[attn.wq_b.weight] = ("_axis_1", None)
        specs[attn.wo_a.weight] = ("_axis_1", None)
        specs[attn.wo_b.weight] = (None, "_axis_1")
        specs[attn.kv_cache] = ("_axis_0", None, None)
        if attn.compress_ratio:
            specs[attn.compressor.kv_cache] = ("_axis_0", None, None)
            specs[attn.compressor.kv_state] = ("_axis_0", None, None)
            specs[attn.compressor.score_state] = ("_axis_0", None, None)
            if attn.indexer is not None:
                specs[attn.indexer.wq_b.weight] = ("_axis_1", None)
                specs[attn.indexer.weights_proj.weight] = ("_axis_1", None)
                specs[attn.indexer.compressor.kv_cache] = ("_axis_0", None, None)
                specs[attn.indexer.compressor.kv_state] = ("_axis_0", None, None)
                specs[attn.indexer.compressor.score_state] = (
                    "_axis_0", None, None,
                )

        # ---- MoE (post-swap) ----
        # block.ffn is A2aSparseMLPWithSharedExperts; .mlp is A2aSparseMLP;
        # .shared_experts is the original Expert.
        a2a_with_shared = block.ffn
        mlp = a2a_with_shared.mlp
        # Replicate gate.weight (sharding it changes fp32 reduction order
        # and flips topk choices; see test_deepseek_v4_flash_moe.py:140-145).
        specs[mlp.router.gate.weight] = (None, None)
        # tid2eid (hash) or bias (non-hash): replicate.

        # Stacked experts: compound shard along both mesh axes.
        specs[mlp.experts.gate_proj] = (compound, None, None)
        specs[mlp.experts.up_proj] = (compound, None, None)
        specs[mlp.experts.down_proj] = (compound, None, None)

        shared = a2a_with_shared.shared_experts
        if shared is not None:
            specs[shared.w1.weight] = (None, None)
            specs[shared.w2.weight] = (None, None)
            specs[shared.w3.weight] = (None, None)

    return specs


# ----------------------------------------------------------------------------
# The test.
# ----------------------------------------------------------------------------


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.parametrize("expert_dtype", ["bfp_bf4", "bfp_bf8"])
@torch.inference_mode()
def test_e2e_prefill_decode_full_real(expert_dtype: str) -> None:
    enable_spmd()
    xr.set_device_type("TT")
    torch.manual_seed(0)

    # --- Mesh + fixed batch ----------------------------------------------
    # bsz=32 is required by A2aSparseMLP's dispatch/combine op; on a (4, 8)
    # mesh that's 8 rows per _axis_0 device so the
    # ("_axis_0", None, None) kv/cache shardings stay valid. The same prompt
    # is tiled across the batch — every row generates the same answer.
    mesh, mesh_shape = make_mesh()
    bsz = BATCH_SIZE
    # assert bsz % mesh_shape[0] == 0, (
    #     f"BATCH_SIZE={bsz} must be divisible by mesh_shape[0]={mesh_shape[0]}"
    # )

    # --- Real config, MTP off, layer-prefix truncation --------------------
    args = weight_loader.load_config_args()
    args.n_mtp_layers = 0
    args.max_batch_size = bsz
    # 128 = window_size = max(compress_ratios). Decode steps wrap into the
    # circular kv_cache modulo this size.
    args.max_seq_len = 128
    if NUM_LAYERS < args.n_layers:
        args.n_layers = NUM_LAYERS
        args.compress_ratios = args.compress_ratios[:NUM_LAYERS]
    print(f"[args] n_layers={args.n_layers}, n_routed_experts={args.n_routed_experts}, "
          f"n_activated_experts={args.n_activated_experts}, "
          f"bsz={bsz}, max_seq_len={args.max_seq_len}, "
          f"compress_ratios={args.compress_ratios}", flush=True)

    # --- Tokenizer + 32 distinct prompts ----------------------------------
    # Each row gets its own prompt, left-padded to PROMPT_LEN so the actual
    # text sits at the right edge (most-recent context is intact for the
    # first decode token). Truncate from the left if a prompt overflows.
    from transformers import AutoTokenizer  # noqa: WPS433

    tokenizer = AutoTokenizer.from_pretrained(weight_loader.REPO_ID)
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    prompt_rows = []
    for prompt in PROMPTS:
        ids = tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids[0]
        if ids.shape[0] >= PROMPT_LEN:
            ids = ids[-PROMPT_LEN:]
        else:
            pad = torch.full(
                (PROMPT_LEN - ids.shape[0],), pad_id, dtype=torch.long
            )
            ids = torch.cat([pad, ids], dim=0)
        prompt_rows.append(ids)
    prompt_ids = torch.stack(prompt_rows, dim=0).contiguous()
    assert prompt_ids.shape == (bsz, PROMPT_LEN)
    print(f"[tokenize] prompt_ids[0][-8:]={prompt_ids[0][-8:].tolist()}", flush=True)

    # --- Build model on CPU, load real weights, swap MoE per layer --------
    model = _build_and_load_full_model(args, mesh_shape)

    # --- Move to TT device ------------------------------------------------
    print("[device] moving model to TT (this releases CPU storage) ...", flush=True)
    device = torch_xla.device()
    model = model.to(device)
    gc.collect()

    for tensor, spec in transformer_shard_spec(model).items():
        xs.mark_sharding(tensor, mesh, spec)

    # --- Apply weight dtype overrides ------------------------------------
    # MoE routed experts + router: bfp4 
    weight_dtype_overrides = {
        # MoE routed experts (compound stacked across the 256 experts).
        "layers.*.ffn.mlp.experts.gate_proj": expert_dtype,
        "layers.*.ffn.mlp.experts.up_proj":   expert_dtype,
        "layers.*.ffn.mlp.experts.down_proj": expert_dtype,
    }
    applied = apply_weight_dtype_overrides(model, weight_dtype_overrides)
    print(f"[wdtype] applied {len(applied)} weight dtype overrides", flush=True)
    for path, dtype_str in applied:
        print(f"[wdtype]   {path} -> {dtype_str}", flush=True)

    hook = sharding_constraint_hook(model.head, mesh, (None, None))
    model.head.register_forward_hook(hook)
    # --- Compile (one handle, reused for prefill and every decode step) --
    # Compiled-mode pattern (mirrors test_prefill_decode_cache_coherence_compiled
    # in test_deepseek_v4_prefill_decode_loop_no_int.py:243-251): pass start_pos
    # as a fresh CPU->device tensor argument every call. dynamo turns it into a
    # symbolic graph input, so its value does NOT constant-fold per step. The
    # lazy-mode sp_buffer/one_buffer trick (lines 142-158 of that same file) is
    # NOT needed here and actively breaks the dynamo bridge's input-update path
    # (Bad StatusOr access during torch_xla.sync).
    #compile_options = {"tt_legacy_compile": True}
    compiled = torch.compile(model, backend="tt")

    # Per-row generation tracker. generated[i] is the list of token ids
    # produced for row i across prefill + every decode step.
    generated: list[list[int]] = [[] for _ in range(bsz)]

    # =================== PREFILL =====================================
    print("[prefill] compiling + running ...", flush=True)
    prompt_ids_tt = prompt_ids.to(device)
    xs.mark_sharding(prompt_ids_tt, mesh, ("_axis_0", None))
    sp_tt = torch.tensor(PROMPT_LEN, dtype=torch.long).to(device)
    prefill_logits = compiled(prompt_ids_tt, sp_tt)  # [bsz, vocab]
    torch_xla.sync(wait=True)
    next_ids = prefill_logits.detach().to("cpu").argmax(dim=-1)  # [bsz]
    for i in range(bsz):
        generated[i].append(int(next_ids[i].item()))
    print(f"[prefill] first ids[:8]={next_ids[:8].tolist()}", flush=True)

    # =================== DECODE LOOP =================================
    # All rows step together; we don't early-exit on per-row EOS so the
    # graph stays shape-stable. EOS trimming happens at print time.
    prev_token = next_ids.unsqueeze(1)  # [bsz, 1]
    for step in range(MAX_NEW_TOKENS - 1):
        start_pos = PROMPT_LEN + step
        prev_token_tt = prev_token.to(device)
        xs.mark_sharding(prev_token_tt, mesh, ("_axis_0", None))
        sp_tt = torch.tensor(start_pos, dtype=torch.long).to(device)
        decode_logits = compiled(prev_token_tt, sp_tt)
        #torch_xla.sync()

        next_ids = decode_logits.detach().to("cpu").argmax(dim=-1)  # [bsz]
        for i in range(bsz):
            generated[i].append(int(next_ids[i].item()))
        print(f"[decode {step + 1:>2}] sp={start_pos}: "
              f"ids[:8]={next_ids[:8].tolist()}", flush=True)
        prev_token = next_ids.unsqueeze(1)

    _print_decoded(tokenizer, PROMPTS, generated)


def _print_decoded(tokenizer, prompts: list, generated: list) -> None:
    eos = tokenizer.eos_token_id
    bar = "=" * 72
    print(f"\n{bar}", flush=True)
    for i, (prompt, ids) in enumerate(zip(prompts, generated)):
        # Trim the per-row continuation at the first EOS (if any) so
        # decoded text doesn't include junk past end-of-sequence.
        cut = next((k for k, t in enumerate(ids) if t == eos), len(ids))
        ids = ids[:cut]
        cont = tokenizer.decode(ids, skip_special_tokens=True)
        print(f"[row {i:02d}] prompt={prompt!r}")
        print(f"         ids={ids}")
        print(f"         cont={cont!r}")
    print(f"{bar}\n", flush=True)
