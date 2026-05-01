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
import math
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

# ============================================================================
# TOY MODEL CONFIG FOR INFRASTRUCTURE TESTING
# Set USE_TOY_MODEL = False and revert other constants for real model runs.
# ============================================================================
USE_TOY_MODEL = True

# Maximum prompt length after tokenization. Prompts longer than this are
# truncated from the left. For test_e2e_prefill_decode_full_real, no padding
# is applied — prompts keep their natural length and the prefill runs up to
# min(prompt_lens). Other tests (PCC tests) still use left-padding to
# PROMPT_LEN for simpler CPU-vs-device comparison.
MAX_PROMPT_LEN = 16 if USE_TOY_MODEL else 128
PROMPT_LEN = MAX_PROMPT_LEN  # Alias for PCC tests that use padding
MAX_NEW_TOKENS = 8 if USE_TOY_MODEL else 64

# Batch size is a hard requirement of the A2aSparseMLP op; on the (4, 8)
# mesh this shards to 8 rows per _axis_0 device. Always 32 regardless of
# toy model mode.
BATCH_SIZE = 32

# Number of transformer blocks to load + run. Real config has 43; truncating
# to a prefix is a memory dial. Layer subset is sliced from
# real_args.compress_ratios so the included layers stay consistent with the
# real-checkpoint layout (mirrors transformer_args in
# test_deepseek_v4_tp_no_int.py:163).
NUM_LAYERS = 3 if USE_TOY_MODEL else 43

# see issue https://github.com/tenstorrent/tt-xla/issues/4444
torch._dynamo.config.cache_size_limit = 100

# One distinct prompt per batch row; len(PROMPTS) must equal BATCH_SIZE.
_PROMPTS_FULL = [
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
# For toy model, we still need 32 prompts to match BATCH_SIZE; just use
# shorter/simpler ones.
PROMPTS = _PROMPTS_FULL
assert len(PROMPTS) == BATCH_SIZE, (
    f"PROMPTS has {len(PROMPTS)} entries, must equal BATCH_SIZE={BATCH_SIZE}"
)


# ----------------------------------------------------------------------------
# Toy model helpers (for infrastructure testing without real weights).
# ----------------------------------------------------------------------------


def _toy_model_args(**overrides) -> mdo.ModelArgs:
    """Small ModelArgs for fast infrastructure testing."""
    defaults = dict(
        max_batch_size=32,  # Must match BATCH_SIZE for 4x8 mesh
        max_seq_len=64,  # Must fit MAX_PROMPT_LEN + MAX_NEW_TOKENS
        vocab_size=129280,  # Keep real vocab_size so tokenizer works
        dim=128,
        moe_inter_dim=64,
        n_layers=3,
        n_mtp_layers=0,
        n_heads=4,
        q_lora_rank=64,
        head_dim=32,
        rope_head_dim=16,
        n_routed_experts=4,
        n_activated_experts=2,
        n_shared_experts=1,
        o_groups=2,
        o_lora_rank=32,
        window_size=8,
        compress_ratios=(0, 4, 8),
        index_n_heads=4,
        index_head_dim=16,
        index_topk=4,
        hc_mult=2,
        hc_sinkhorn_iters=3,
    )
    defaults.update(overrides)
    return mdo.ModelArgs(**defaults)


def _init_weights(module: torch.nn.Module, std: float = 0.02) -> None:
    """Initialize all parameters with a normal distribution."""
    from third_party.tt_forge_models.deepseek_v4.modified_model.model_decode_opt import (
        RMSNorm,
    )
    with torch.no_grad():
        for sub in module.modules():
            if isinstance(sub, RMSNorm):
                continue
            for _, param in sub.named_parameters(recurse=False):
                torch.nn.init.normal_(param, mean=0.0, std=std)


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

    When USE_TOY_MODEL is True, skips real weight loading and uses random
    initialization instead.
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

    if USE_TOY_MODEL:
        print("[load] USE_TOY_MODEL=True, using random weight initialization ...", flush=True)
        _init_weights(model)
    else:
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

    # --- Model config, MTP off, layer-prefix truncation --------------------
    if USE_TOY_MODEL:
        args = _toy_model_args()
        print("[config] USE_TOY_MODEL=True, using toy model config", flush=True)
    else:
        args = weight_loader.load_config_args()
        args.n_mtp_layers = 0
        args.max_batch_size = bsz
        if NUM_LAYERS < args.n_layers:
            args.n_layers = NUM_LAYERS
            args.compress_ratios = args.compress_ratios[:NUM_LAYERS]
        # max_seq_len must cover both the longest prompt (up to MAX_PROMPT_LEN)
        # and all decode positions. Round up to the next multiple of the largest
        # compress_ratio so every compressor's kv_cache has room for at least one
        # extra compressed slot beyond the prefill fill.
        max_compress_ratio = max(args.compress_ratios) if any(args.compress_ratios) else 1
        args.max_seq_len = math.ceil((MAX_PROMPT_LEN + MAX_NEW_TOKENS) / max_compress_ratio) * max_compress_ratio
    print(f"[args] n_layers={args.n_layers}, n_routed_experts={args.n_routed_experts}, "
          f"n_activated_experts={args.n_activated_experts}, "
          f"bsz={bsz}, max_seq_len={args.max_seq_len}, "
          f"compress_ratios={args.compress_ratios}", flush=True)

    # --- Tokenizer + 32 distinct prompts ----------------------------------
    # Each row gets its own prompt with no padding. Prompts are truncated
    # from the left if they exceed MAX_PROMPT_LEN. The generation loop
    # prefills up to the shortest prompt and then generates tokens,
    # overriding model predictions with ground-truth tokens for positions
    # still within each row's prompt.
    from transformers import AutoTokenizer  # noqa: WPS433

    tokenizer = AutoTokenizer.from_pretrained(weight_loader.REPO_ID)
    eos_id = tokenizer.eos_token_id
    prompt_tokens: list[list[int]] = []
    for prompt in PROMPTS:
        ids = tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids[0].tolist()
        if len(ids) > MAX_PROMPT_LEN:
            ids = ids[-MAX_PROMPT_LEN:]
        prompt_tokens.append(ids)

    prompt_lens = [len(t) for t in prompt_tokens]
    min_prompt_len = min(prompt_lens)
    max_prompt_len = max(prompt_lens)
    total_len = min(args.max_seq_len, MAX_NEW_TOKENS + max_prompt_len)

    # Static output tensor: fill with -1, then insert each prompt's tokens.
    tokens = torch.full((bsz, total_len), -1, dtype=torch.long)
    for i, t in enumerate(prompt_tokens):
        tokens[i, : len(t)] = torch.tensor(t, dtype=torch.long)
    prompt_mask = tokens != -1  # True where tokens are from the prompt

    print(
        f"[tokenize] prompt_lens={prompt_lens[:8]}... min={min_prompt_len} "
        f"max={max_prompt_len} total_len={total_len}",
        flush=True,
    )

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
        # Router gate: keep at bf16 (quantizing to bfp4/8 flips topk choices).
        "layers.*.ffn.mlp.router.gate.weight": "bf16",
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

    prev_pos = 0

    # =================== PREFILL + DECODE LOOP =======================
    # The first forward pass processes [0:min_prompt_len] tokens (prefill).
    # Subsequent passes generate one token at a time (decode). For positions
    # still within a row's prompt, the ground-truth token overrides the
    # model's prediction.
    #
    # Note: The model does not support batch size < max_batch_size, so we
    # continue decoding for the full batch even after rows hit EOS. EOS
    # trimming happens at output extraction time.
    for cur_pos in range(min_prompt_len, total_len):
        input_tokens = tokens[:, prev_pos:cur_pos]
        input_tokens_tt = input_tokens.to(device)
        xs.mark_sharding(input_tokens_tt, mesh, ("_axis_0", None))
        sp_tt = torch.tensor(cur_pos, dtype=torch.long).to(device)

        if prev_pos == 0:
            print(
                f"[prefill] compiling + running (positions 0:{cur_pos}) ...",
                flush=True,
            )
        else:
            print(
                f"[decode] cur_pos={cur_pos} prev_pos={prev_pos}",
                flush=True,
            )

        logits = compiled(input_tokens_tt, sp_tt)  # [bsz, vocab]
        torch_xla.sync(wait=True)

        # Sample next token (greedy argmax).
        next_token = logits.detach().to("cpu").argmax(dim=-1)  # [bsz]

        # For positions still within a row's prompt, override with ground
        # truth so the model sees the correct context.
        next_token = torch.where(
            prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token
        )
        tokens[:, cur_pos] = next_token

        if prev_pos == 0:
            print(
                f"[prefill] done. next_ids[:8]={next_token[:8].tolist()}",
                flush=True,
            )
        else:
            print(
                f"[decode] cur_pos={cur_pos}: ids[:8]={next_token[:8].tolist()}",
                flush=True,
            )

        prev_pos = cur_pos

    # Extract completion tokens (everything after the prompt).
    completion_tokens: list[list[int]] = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i] : prompt_lens[i] + MAX_NEW_TOKENS]
        if eos_id in toks:
            toks = toks[: toks.index(eos_id)]
        completion_tokens.append(toks)

    _print_decoded(tokenizer, PROMPTS, completion_tokens)


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


# ----------------------------------------------------------------------------
# CPU-vs-device prefill PCC test.
# ----------------------------------------------------------------------------


PREFILL_PCC_REQUIRED = 0.95


def _pcc(x: torch.Tensor, y: torch.Tensor) -> float:
    """Pearson correlation. Mirrors _pcc in
    test_deepseek_v4_prefill_decode_loop_no_int.py: float64 conversion,
    allclose short-circuit (PCC is ill-conditioned when tensors are nearly
    identical), Pearson on flattened tensors."""
    x = x.detach().to(torch.float64).flatten()
    y = y.detach().to(torch.float64).flatten()
    if torch.allclose(x, y, rtol=1e-2, atol=1e-2):
        return 1.0
    if x.numel() <= 1:
        return 0.0
    vx = x - x.mean()
    vy = y - y.mean()
    denom = vx.norm() * vy.norm()
    if denom == 0:
        return float("nan")
    return float((vx @ vy) / denom)


def _reset_attn_caches(model: mdo.Transformer) -> None:
    """Zero in-place attention/compressor cache buffers populated by a prior
    forward pass so the next prefill starts from the same blank state as a
    fresh model. Mirrors the buffer set asserted in transformer_shard_spec."""
    for block in model.layers:
        attn = block.attn
        attn.kv_cache.zero_()
        if attn.compress_ratio:
            attn.compressor.kv_cache.zero_()
            attn.compressor.kv_state.zero_()
            attn.compressor.score_state.fill_(float("-inf"))
            if attn.indexer is not None:
                attn.indexer.compressor.kv_cache.zero_()
                attn.indexer.compressor.kv_state.zero_()
                attn.indexer.compressor.score_state.fill_(float("-inf"))


def _emulate_post_prefill_caches(
    model: mdo.Transformer, generator: torch.Generator, std: float = 0.1
) -> None:
    """Fill kv / compressor / indexer cache buffers with deterministic
    random values, emulating a non-empty post-prefill state. Same generator
    seed -> identical fills, so a CPU run and a device run can share bit-
    identical starting cache state without paying for a real CPU prefill.

    score_state would normally hold a mix of real scores and -inf for slots
    not yet computed; the model only uses it via softmax(dim=1) so finite
    random values are mathematically valid here (compressor logic at
    model_decode_opt.py:414,424). std=0.1 keeps softmax in-distribution."""
    for block in model.layers:
        attn = block.attn
        attn.kv_cache.normal_(mean=0.0, std=std, generator=generator)
        if attn.compress_ratio:
            attn.compressor.kv_cache.normal_(mean=0.0, std=std, generator=generator)
            attn.compressor.kv_state.normal_(mean=0.0, std=std, generator=generator)
            attn.compressor.score_state.normal_(mean=0.0, std=std, generator=generator)
            if attn.indexer is not None:
                attn.indexer.compressor.kv_cache.normal_(
                    mean=0.0, std=std, generator=generator
                )
                attn.indexer.compressor.kv_state.normal_(
                    mean=0.0, std=std, generator=generator
                )
                attn.indexer.compressor.score_state.normal_(
                    mean=0.0, std=std, generator=generator
                )


def _tokenize_prompts(tokenizer, prompts: list) -> torch.Tensor:
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    prompt_rows = []
    for prompt in prompts:
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
    return torch.stack(prompt_rows, dim=0).contiguous()


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.parametrize("num_layers", [10, 15, 20, 30, 43])
@torch.inference_mode()
def test_e2e_prefill_pcc(num_layers: int) -> None:
    """Run prefill on CPU and on the TT device with a `num_layers`-deep
    swapped Transformer and PCC-compare the final hidden-states output.

    Both paths share one model object: the A2aSparseMLP swap routes through
    `_cpu_forward` -> original MoE while `hidden_states.device.type == "cpu"`
    (sparse_mlp.py:540) and through the all_to_all dispatch/combine kernels
    once moved to the TT device. CPU prefill runs first; its in-place kv /
    compressor / indexer cache mutations are zeroed before the device move
    so the device prefill starts from the same blank state.

    The "final output hidden states" tensor compared here is the post-block,
    pre-head activation `h` of shape `[bsz, seq, hc_mult, dim]` — the first
    element of the tuple returned by
    `Transformer.forward(..., return_hidden_states=True)`
    (model_decode_opt.py:972-981).
    """
    enable_spmd()
    xr.set_device_type("TT")
    torch.manual_seed(0)

    mesh, mesh_shape = make_mesh()
    bsz = BATCH_SIZE

    args = weight_loader.load_config_args()
    args.n_mtp_layers = 0
    args.max_batch_size = bsz
    args.max_seq_len = 128
    if num_layers < args.n_layers:
        args.n_layers = num_layers
        args.compress_ratios = args.compress_ratios[:num_layers]
    print(
        f"[args] n_layers={args.n_layers}, "
        f"n_routed_experts={args.n_routed_experts}, "
        f"n_activated_experts={args.n_activated_experts}, bsz={bsz}, "
        f"max_seq_len={args.max_seq_len}, "
        f"compress_ratios={args.compress_ratios}",
        flush=True,
    )

    from transformers import AutoTokenizer  # noqa: WPS433

    tokenizer = AutoTokenizer.from_pretrained(weight_loader.REPO_ID)
    prompt_ids = _tokenize_prompts(tokenizer, PROMPTS)
    assert prompt_ids.shape == (bsz, PROMPT_LEN)
    print(
        f"[tokenize] prompt_ids[0][-8:]={prompt_ids[0][-8:].tolist()}",
        flush=True,
    )

    model = _build_and_load_full_model(args, mesh_shape)

    # --- CPU prefill ------------------------------------------------------
    print("[cpu] prefill ...", flush=True)
    sp_cpu = torch.tensor(PROMPT_LEN, dtype=torch.long)
    cpu_h, _ = model(prompt_ids, sp_cpu, return_hidden_states=True)
    cpu_out = cpu_h.detach().to(torch.float32).cpu().clone()
    print(f"[cpu] hidden_states shape={tuple(cpu_out.shape)}", flush=True)

    # CPU prefill mutated kv_cache / kv_state / score_state in place. Reset
    # them so the device prefill sees the same fresh-model starting state.
    _reset_attn_caches(model)

    # --- Move to device + shard ------------------------------------------
    print("[device] moving model to TT (this releases CPU storage) ...", flush=True)
    device = torch_xla.device()
    model = model.to(device)
    gc.collect()
    for tensor, spec in transformer_shard_spec(model).items():
        xs.mark_sharding(tensor, mesh, spec)

    # head is computed even under return_hidden_states=True (logits are the
    # second tuple element, ignored here) — re-attach the sharding hook so
    # head's output carries the standard `(None, None)` constraint, matching
    # test_e2e_prefill_decode_full_real.
    hook = sharding_constraint_hook(model.head, mesh, (None, None))
    model.head.register_forward_hook(hook)

    compiled = torch.compile(model, backend="tt")

    prompt_ids_tt = prompt_ids.to(device)
    xs.mark_sharding(prompt_ids_tt, mesh, ("_axis_0", None))
    sp_tt = torch.tensor(PROMPT_LEN, dtype=torch.long).to(device)

    # --- Device prefill ---------------------------------------------------
    print("[device] compiling + running prefill ...", flush=True)
    device_h, _ = compiled(prompt_ids_tt, sp_tt, return_hidden_states=True)
    torch_xla.sync(wait=True)
    device_out = device_h.detach().to("cpu").to(torch.float32).clone()
    print(f"[device] hidden_states shape={tuple(device_out.shape)}", flush=True)

    assert cpu_out.shape == device_out.shape, (
        f"shape mismatch: cpu={tuple(cpu_out.shape)} "
        f"device={tuple(device_out.shape)}"
    )

    pcc = _pcc(cpu_out, device_out)
    print(
        f"[pcc] cpu vs device prefill hidden_states pcc={pcc:.6f} "
        f"(required >= {PREFILL_PCC_REQUIRED})",
        flush=True,
    )
    assert pcc >= PREFILL_PCC_REQUIRED, (
        f"PCC failed: got {pcc:.6f}, required >= {PREFILL_PCC_REQUIRED}"
    )


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.parametrize("num_layers", [10, 15, 20, 30, 43])
@torch.inference_mode()
def test_e2e_single_decode_pcc(num_layers: int) -> None:
    """Run one decode step (seqlen=1) on CPU and on the TT device starting
    from a deterministically-emulated post-prefill cache state, and
    PCC-compare the post-block, pre-head hidden states.

    Strategy: build one swapped Transformer on CPU. Fill kv / compressor /
    indexer caches with random values from a seeded generator, run a single
    CPU decode at start_pos=PROMPT_LEN. Re-fill the caches with the same
    seed (CPU decode mutated them in place), then move the model to the TT
    device — `.to(device)` carries the freshly re-emulated cache contents
    over so the device decode sees bit-identical starting state. Run a
    single device decode and compare.

    The "final hidden states" tensor compared here is the post-block,
    pre-head activation `h` returned by
    `Transformer.forward(..., return_hidden_states=True)`
    (model_decode_opt.py:972-981).

    Real prefill is not run because a 43-layer / 256-expert CPU prefill of
    a 128-token prompt is prohibitively slow; this test isolates decode-
    forward equivalence — what matters is that both sides start the decode
    step from identical caches.
    """
    enable_spmd()
    xr.set_device_type("TT")
    torch.manual_seed(0)

    mesh, mesh_shape = make_mesh()
    bsz = BATCH_SIZE

    args = weight_loader.load_config_args()
    args.n_mtp_layers = 0
    args.max_batch_size = bsz
    # max_seq_len > PROMPT_LEN: a decode at start_pos = PROMPT_LEN reads
    # `freqs_cis[start_pos]` directly (model_decode_opt.py:644), so the
    # precomputed table needs at least one entry past PROMPT_LEN. The
    # prefill+decode test gets away with max_seq_len = PROMPT_LEN = 128
    # because it only exercises decode under torch.compile on device, where
    # an OOB index_select doesn't raise the same way it does on CPU.
    args.max_seq_len = 2 * PROMPT_LEN
    if num_layers < args.n_layers:
        args.n_layers = num_layers
        args.compress_ratios = args.compress_ratios[:num_layers]
    print(
        f"[args] n_layers={args.n_layers}, "
        f"n_routed_experts={args.n_routed_experts}, "
        f"n_activated_experts={args.n_activated_experts}, bsz={bsz}, "
        f"max_seq_len={args.max_seq_len}, "
        f"compress_ratios={args.compress_ratios}",
        flush=True,
    )

    model = _build_and_load_full_model(args, mesh_shape)

    # Single-token decode input: id 0 per batch row. Deterministic and the
    # actual content doesn't affect the CPU-vs-device equivalence check.
    next_token = torch.zeros((bsz, 1), dtype=torch.long)
    cache_seed = 12345

    # --- CPU decode -------------------------------------------------------
    _emulate_post_prefill_caches(model, torch.Generator().manual_seed(cache_seed))
    print("[cpu] single decode (start_pos=PROMPT_LEN) ...", flush=True)
    sp_cpu = torch.tensor(PROMPT_LEN, dtype=torch.long)
    cpu_h, _ = model(next_token, sp_cpu, return_hidden_states=True)
    cpu_out = cpu_h.detach().to(torch.float32).cpu().clone()
    print(f"[cpu] hidden_states shape={tuple(cpu_out.shape)}", flush=True)

    # CPU decode mutated kv_cache + compressor / indexer state in place.
    # Re-emulate from the same seed so the device run starts from identical
    # cache contents.
    _emulate_post_prefill_caches(model, torch.Generator().manual_seed(cache_seed))

    # --- Move to device + shard ------------------------------------------
    print("[device] moving model to TT (this releases CPU storage) ...", flush=True)
    device = torch_xla.device()
    model = model.to(device)
    gc.collect()
    for tensor, spec in transformer_shard_spec(model).items():
        xs.mark_sharding(tensor, mesh, spec)

    hook = sharding_constraint_hook(model.head, mesh, (None, None))
    model.head.register_forward_hook(hook)

    compiled = torch.compile(model, backend="tt")

    next_token_tt = next_token.to(device)
    xs.mark_sharding(next_token_tt, mesh, ("_axis_0", None))
    sp_tt = torch.tensor(PROMPT_LEN, dtype=torch.long).to(device)

    # --- Device decode ----------------------------------------------------
    print("[device] compiling + running single decode ...", flush=True)
    device_h, _ = compiled(next_token_tt, sp_tt, return_hidden_states=True)
    torch_xla.sync(wait=True)
    device_out = device_h.detach().to("cpu").to(torch.float32).clone()
    print(f"[device] hidden_states shape={tuple(device_out.shape)}", flush=True)

    assert cpu_out.shape == device_out.shape, (
        f"shape mismatch: cpu={tuple(cpu_out.shape)} "
        f"device={tuple(device_out.shape)}"
    )

    pcc = _pcc(cpu_out, device_out)
    print(
        f"[pcc] cpu vs device single-decode hidden_states pcc={pcc:.6f} "
        f"(required >= {PREFILL_PCC_REQUIRED})",
        flush=True,
    )
    assert pcc >= PREFILL_PCC_REQUIRED, (
        f"PCC failed: got {pcc:.6f}, required >= {PREFILL_PCC_REQUIRED}"
    )


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.parametrize("num_layers", [10, 15, 20, 30, 43])
@pytest.mark.parametrize("num_iterations", [1, 2, 5, 10])
@pytest.mark.parametrize("use_cpu_decode_inputs", [True, False])
@torch.inference_mode()
def test_prefill_and_decode_pcc(
    num_layers: int,
    num_iterations: int,
    use_cpu_decode_inputs: bool,
) -> None:
    """Run prefill + `num_iterations` decode steps on CPU, then on the TT
    device, capturing the post-block hidden states tensor at every step.
    PCC-compare CPU vs device for the prefill output and for each decode
    step.

    Decode is driven by real logits: the first decode token comes from
    argmax of the prefill logits, and each subsequent decode token comes
    from argmax of the previous decode's logits — autoregressive
    generation.

    `use_cpu_decode_inputs` selects how the device decode loop is fed:
      - True: the device replays the CPU-derived token sequence verbatim,
        so CPU and device see identical inputs at every decode step and
        the per-step PCC isolates numeric divergence rather than diverging
        argmax choices. This is the path whose decode-step PCCs are
        asserted.
      - False: the device derives its own next-token sequence from its
        own logits. At the end of every decode step we record the per-row
        match count between the token the device just argmaxed and the
        token CPU argmaxed at the same step, and print the list at the
        end (no assert — the user wants to inspect divergence). Decode-
        step PCCs are still computed and printed but not asserted, since
        once argmaxes diverge the inputs to subsequent steps differ and
        decode-PCC stops being equivalence-meaningful.

    `Transformer.forward(..., return_hidden_states=True)` returns
    `(hidden_states, logits)` (model_decode_opt.py:972-981). hidden_states
    is `[bsz, seqlen, hc_mult, dim]` (seqlen=PROMPT_LEN for prefill, 1 for
    each decode); logits is `[bsz, vocab_size]` at the last position.

    Caches mutate in place across prefill+decodes; CPU caches are zeroed
    before the device move so the device run starts from a fresh-model
    state matching the CPU run.
    """
    enable_spmd()
    xr.set_device_type("TT")
    torch.manual_seed(0)

    mesh, mesh_shape = make_mesh()
    bsz = BATCH_SIZE

    args = weight_loader.load_config_args()
    args.n_mtp_layers = 0
    args.max_batch_size = bsz
    # Extend the max_seq_len to allow kv_cache tensors to store all prefill+decode tokens
    args.max_seq_len = 2 * PROMPT_LEN
    assert PROMPT_LEN + num_iterations - 1 < args.max_seq_len, (
        f"PROMPT_LEN+num_iterations-1 ({PROMPT_LEN + num_iterations - 1}) "
        f">= max_seq_len ({args.max_seq_len})"
    )
    if num_layers < args.n_layers:
        args.n_layers = num_layers
        args.compress_ratios = args.compress_ratios[:num_layers]
    print(
        f"[args] n_layers={args.n_layers}, "
        f"n_routed_experts={args.n_routed_experts}, "
        f"n_activated_experts={args.n_activated_experts}, bsz={bsz}, "
        f"max_seq_len={args.max_seq_len}, num_iterations={num_iterations}, "
        f"compress_ratios={args.compress_ratios}",
        flush=True,
    )

    from transformers import AutoTokenizer  # noqa: WPS433

    tokenizer = AutoTokenizer.from_pretrained(weight_loader.REPO_ID)
    prompt_ids = _tokenize_prompts(tokenizer, PROMPTS)
    assert prompt_ids.shape == (bsz, PROMPT_LEN)

    model = _build_and_load_full_model(args, mesh_shape)

    # ---- CPU prefill + decodes ------------------------------------------
    # decode_inputs[k] is the [bsz, 1] token tensor fed into CPU decode
    # step k (decode_inputs[0] is argmax of prefill logits;
    # decode_inputs[k>0] is argmax of decode step k-1's logits).
    # cpu_generated_tokens[k] is the token CPU produced *at* decode step k
    # (= argmax of step k's logits = decode_inputs[k+1] when k+1 exists).
    decode_inputs: list[torch.Tensor] = []
    cpu_generated_tokens: list[torch.Tensor] = []

    print("[cpu] prefill ...", flush=True)
    sp_prefill = torch.tensor(PROMPT_LEN, dtype=torch.long)
    # h_pf is the hidden states tensor returned after prefill
    # logits_pf is the logits tensor returned after prefill
    h_pf, logits_pf = model(prompt_ids, sp_prefill, return_hidden_states=True)
    cpu_prefill_h = h_pf.detach().to(torch.float32).cpu().clone()
    next_token = logits_pf.detach().cpu().argmax(dim=-1, keepdim=True).to(torch.long)
    print(
        f"[cpu] prefill hidden_states shape={tuple(cpu_prefill_h.shape)} "
        f"first_ids[:8]={next_token[:8, 0].tolist()}",
        flush=True,
    )

    cpu_decode_hs: list[torch.Tensor] = []
    for step in range(num_iterations):
        decode_inputs.append(next_token)
        sp_step = torch.tensor(PROMPT_LEN + step, dtype=torch.long)
        h_d, logits_d = model(next_token, sp_step, return_hidden_states=True)
        h = h_d.detach().to(torch.float32).cpu().clone()
        cpu_decode_hs.append(h)
        next_token = logits_d.detach().cpu().argmax(dim=-1, keepdim=True).to(torch.long)
        cpu_generated_tokens.append(next_token)
        print(
            f"[cpu] decode step {step} sp={PROMPT_LEN + step} "
            f"shape={tuple(h.shape)} next_ids[:8]={next_token[:8, 0].tolist()}",
            flush=True,
        )

    assert len(decode_inputs) == num_iterations
    assert len(cpu_generated_tokens) == num_iterations

    # CPU prefill+decodes mutated kv_cache + compressor / indexer state in
    # place. Reset so the device run starts from the same fresh state.
    _reset_attn_caches(model)

    # ---- Move to device + shard ----------------------------------------
    print("[device] moving model to TT (this releases CPU storage) ...", flush=True)
    device = torch_xla.device()
    model = model.to(device)
    gc.collect()
    for tensor, spec in transformer_shard_spec(model).items():
        xs.mark_sharding(tensor, mesh, spec)

    hook = sharding_constraint_hook(model.head, mesh, (None, None))
    model.head.register_forward_hook(hook)

    compiled = torch.compile(model, backend="tt")

    # ---- Device prefill + decodes --------------------------------------
    prompt_ids_tt = prompt_ids.to(device)
    xs.mark_sharding(prompt_ids_tt, mesh, ("_axis_0", None))
    sp_prefill_tt = torch.tensor(PROMPT_LEN, dtype=torch.long).to(device)
    print("[device] compiling + running prefill ...", flush=True)
    h_pf_dev, logits_pf_dev = compiled(
        prompt_ids_tt, sp_prefill_tt, return_hidden_states=True
    )
    torch_xla.sync(wait=True)
    device_prefill_h = h_pf_dev.detach().to("cpu").to(torch.float32).clone()
    # Device's first decode input when running self-driven (False): the
    # argmax of device's own prefill logits. Materialize now via .to("cpu")
    # so the lazy-graph result for prefill is snapshotted before the
    # decode loop mutates caches.
    next_token_dev = (
        logits_pf_dev.detach().to("cpu").argmax(dim=-1, keepdim=True).to(torch.long)
    )
    print(
        f"[device] prefill hidden_states shape={tuple(device_prefill_h.shape)} "
        f"first_ids[:8]={next_token_dev[:8, 0].tolist()}",
        flush=True,
    )

    device_decode_hs: list[torch.Tensor] = []
    # When use_cpu_decode_inputs=False, store per-step (matches, total) of
    # device-vs-cpu argmax tokens. Empty in the True path.
    token_match_results: list[tuple[int, int, int]] = []

    for step in range(num_iterations):
        if use_cpu_decode_inputs:
            # Replay the CPU-derived token sequence so CPU and device
            # decode steps see identical inputs.
            token_in = decode_inputs[step]
        else:
            # Self-driven: feed device its own previous-step argmax.
            token_in = next_token_dev

        decode_token_tt = token_in.to(device)
        xs.mark_sharding(decode_token_tt, mesh, ("_axis_0", None))
        sp_step_tt = torch.tensor(PROMPT_LEN + step, dtype=torch.long).to(device)
        h_d_dev, logits_d_dev = compiled(
            decode_token_tt, sp_step_tt, return_hidden_states=True
        )
        # `.to("cpu")` forces lazy-graph materialization, snapshotting the
        # i-th decode's hidden states before the next step mutates caches.
        h = h_d_dev.detach().to("cpu").to(torch.float32).clone()
        device_decode_hs.append(h)
        dev_argmax = (
            logits_d_dev.detach().to("cpu").argmax(dim=-1, keepdim=True).to(torch.long)
        )
        next_token_dev = dev_argmax  # used as next-step input when not replaying CPU

        if not use_cpu_decode_inputs:
            matches = int((dev_argmax == cpu_generated_tokens[step]).sum().item())
            token_match_results.append((step, matches, bsz))

        print(
            f"[device] decode step {step} sp={PROMPT_LEN + step} "
            f"shape={tuple(h.shape)} "
            f"next_ids[:8]={dev_argmax[:8, 0].tolist()}",
            flush=True,
        )

    # ---- PCC comparisons ------------------------------------------------
    pccs: list[tuple[str, float]] = []

    assert cpu_prefill_h.shape == device_prefill_h.shape, (
        f"prefill shape mismatch: cpu={tuple(cpu_prefill_h.shape)} "
        f"device={tuple(device_prefill_h.shape)}"
    )
    pcc_prefill = _pcc(cpu_prefill_h, device_prefill_h)
    pccs.append(("prefill", pcc_prefill))
    print(f"[pcc] prefill pcc={pcc_prefill:.6f}", flush=True)

    decode_pccs: list[tuple[str, float]] = []
    for i, (cpu_h, dev_h) in enumerate(zip(cpu_decode_hs, device_decode_hs)):
        assert cpu_h.shape == dev_h.shape, (
            f"decode[{i}] shape mismatch: cpu={tuple(cpu_h.shape)} "
            f"device={tuple(dev_h.shape)}"
        )
        p = _pcc(cpu_h, dev_h)
        decode_pccs.append((f"decode[{i}]", p))
        print(f"[pcc] decode step {i} pcc={p:.6f}", flush=True)

    # ---- Token-sequence match (False-path only) ------------------------
    if not use_cpu_decode_inputs:
        print(
            "[match] device-vs-cpu argmax tokens per decode step "
            f"(use_cpu_decode_inputs=False, total={bsz} rows):",
            flush=True,
        )
        for step, matches, total in token_match_results:
            print(
                f"  step {step}: {matches}/{total} rows match",
                flush=True,
            )

    # ---- Assertions ----------------------------------------------------
    # Prefill PCC is always asserted (prefill inputs are identical for both
    # paths). Decode PCCs are asserted only when use_cpu_decode_inputs=True
    # — once self-driven argmaxes diverge, decode-step inputs differ between
    # CPU and device and the PCC stops being equivalence-meaningful.
    failures = []
    if pcc_prefill < PREFILL_PCC_REQUIRED:
        failures.append(f"prefill={pcc_prefill:.6f}")
    if use_cpu_decode_inputs:
        for label, p in decode_pccs:
            if p < PREFILL_PCC_REQUIRED:
                failures.append(f"{label}={p:.6f}")
    assert not failures, (
        f"PCC failed for: {failures} (required >= {PREFILL_PCC_REQUIRED})"
    )
