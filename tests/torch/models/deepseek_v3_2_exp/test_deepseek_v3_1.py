# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek v3.1 tests on TT (Galaxy 4x8).

Two end-to-end tests, both hand-orchestrated (no dependency on
``run_graph_test``), so they can build on top of ``origin/main`` without
tester-infrastructure changes:

- ``test_deepseek_v3_1_full_sparse_moe``: prefill-only, A2aSparseMLP, legacy
  int-path cache, CPU PCC comparison.
- ``test_deepseek_v3_1_decode_static_cache``: autoregressive greedy decode
  through MLACache (``cache_position`` + ``attention_mask``), 2 compiles
  (prefill + decode), per-step PCC against a CPU int-path reference.

Both share the weight-loading / dequantization / sparse-cache infrastructure
below.
"""
import json
import os
import sys
import time

import numpy as np
import pytest
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from huggingface_hub import hf_hub_download
from infra.testers.compiler_config import CompilerConfig
from modified_model import ModelArgs
from modified_model import Transformer as ModifiedTransformer
from modified_model import precompute_freqs_cis
from safetensors.torch import load_file as safetensors_load_file
from torch import nn
from torch_xla.distributed.spmd import Mesh
from tt_torch.sparse_mlp import A2aSparseMLP, enable_sparse_mlp

# Import shard-streaming + cache helpers from the builder script (single source of truth)
sys.path.insert(0, os.path.dirname(__file__))
from build_weight_cache import (
    _dequant_cache_dir,
    _has_cache,
    _post_sparse_cache_dir,
    build_cache,
    build_post_sparse_cache,
)

DEEPSEEK_V3_1_REPO = "deepseek-ai/DeepSeek-V3.1"


# ---------------------------------------------------------------------------
# Weight loading / cache
# ---------------------------------------------------------------------------


def load_deepseek_config(repo_id=DEEPSEEK_V3_1_REPO):
    """Download and parse the HuggingFace config.json into ModelArgs fields."""
    config_path = hf_hub_download(repo_id, "config.json")
    with open(config_path) as f:
        hf_cfg = json.load(f)
    rope_scaling = hf_cfg["rope_scaling"]
    return ModelArgs(
        vocab_size=hf_cfg["vocab_size"],
        dim=hf_cfg["hidden_size"],
        inter_dim=hf_cfg["intermediate_size"],
        moe_inter_dim=hf_cfg["moe_intermediate_size"],
        n_layers=hf_cfg["num_hidden_layers"],
        n_dense_layers=hf_cfg["first_k_dense_replace"],
        n_heads=hf_cfg["num_attention_heads"],
        n_routed_experts=hf_cfg["n_routed_experts"],
        n_shared_experts=hf_cfg["n_shared_experts"],
        n_activated_experts=hf_cfg["num_experts_per_tok"],
        n_expert_groups=hf_cfg["n_group"],
        n_limited_groups=hf_cfg["topk_group"],
        score_func=hf_cfg["scoring_func"],
        route_scale=hf_cfg["routed_scaling_factor"],
        q_lora_rank=hf_cfg["q_lora_rank"],
        kv_lora_rank=hf_cfg["kv_lora_rank"],
        qk_nope_head_dim=hf_cfg["qk_nope_head_dim"],
        qk_rope_head_dim=hf_cfg["qk_rope_head_dim"],
        v_head_dim=hf_cfg["v_head_dim"],
        original_seq_len=rope_scaling["original_max_position_embeddings"],
        rope_theta=hf_cfg["rope_theta"],
        rope_factor=rope_scaling["factor"],
        beta_fast=rope_scaling["beta_fast"],
        beta_slow=rope_scaling["beta_slow"],
        mscale=rope_scaling["mscale"],
        # index_* keys are V3.2-exp only (sparse-attention indexer); absent in
        # V3.1. Defaults here correspond to the indexer being disabled.
        index_n_heads=hf_cfg.get("index_n_heads", 0),
        index_head_dim=hf_cfg.get("index_head_dim", 128),
        index_topk=hf_cfg.get("index_topk", 2048),
    )


def _load_cache_chunked(cache_dir):
    """Load all chunk files via mmap. Returns merged state_dict."""
    state_dict = {}
    for fname in sorted(os.listdir(cache_dir)):
        if not fname.endswith(".safetensors"):
            continue
        chunk = safetensors_load_file(os.path.join(cache_dir, fname))
        state_dict.update(chunk)
    return state_dict


def _load_modified_dequantized_weights(model, repo_id, n_layers, n_dense_layers=1):
    """Load the dequant cache into ``model``, self-healing it from FP8 shards if absent."""
    cache_dir = _dequant_cache_dir(repo_id, n_layers)
    if not _has_cache(cache_dir):
        print(
            f"[weights] no cache at {cache_dir} — dequantizing from FP8...", flush=True
        )
        build_cache(repo_id, n_layers, n_dense_layers)

    t0 = time.perf_counter()
    print(f"[weights] loading cached BF16 weights from {cache_dir}", flush=True)
    state_dict = _load_cache_chunked(cache_dir)
    print(
        f"[timing] cache load (mmap): {time.perf_counter() - t0:.1f}s "
        f"({len(state_dict)} tensors)",
        flush=True,
    )

    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    print(
        f"[weights] loaded {len(state_dict)} tensors. "
        f"Missing: {len(missing)}, Unexpected: {len(unexpected)}"
    )


def _fix_meta_buffers(model, args):
    """Replace meta-device buffers with properly-computed CPU tensors.

    After meta construction + ``load_state_dict(assign=True)``, non-persistent
    buffers remain on meta. Recompute them on CPU.
    """
    import scipy.linalg

    freqs_cis_complex = precompute_freqs_cis(args)
    model.freqs_cis = torch.view_as_real(freqs_cis_complex)
    hadamard = torch.tensor(
        scipy.linalg.hadamard(args.index_head_dim), dtype=torch.bfloat16
    ) * (args.index_head_dim**-0.5)
    model.hadamard_matrix = hadamard
    # Re-materialize the MLA cache layers on CPU so they're real tensors (not
    # meta) before mark_sharding / .to(device). Each layer holds its own
    # compressed_kv / k_pe buffers, already initialized in Transformer
    # __init__ — but if the model was built on meta, those buffers landed on
    # meta too. Re-run early_initialization directly on CPU.
    mla0 = model.layers[0].attn
    cache_dtype = mla0.wkv_a.weight.dtype
    for cache_layer in model._cache_layers:
        cache_layer.is_initialized = False
        cache_layer.early_initialization(
            batch_size=args.max_batch_size,
            kv_lora_rank=mla0.kv_lora_rank,
            pe_rank=mla0.qk_rope_head_dim,
            dtype=cache_dtype,
            device=torch.device("cpu"),
        )
    for layer in model.layers:
        attn = layer.attn
        attn.hadamard_matrix = hadamard
        if attn.indexer is not None:
            attn.indexer.k_cache = torch.zeros(
                args.max_batch_size,
                args.max_seq_len,
                attn.indexer.head_dim,
                dtype=torch.bfloat16,
            )


def _build_prefill_attention_mask(
    batch_size,
    prompt_len,
    max_cache_len,
    token_attention_mask=None,
    dtype=torch.bfloat16,
):
    """(B, 1, prompt_len, max_cache_len) additive causal mask.

    If ``token_attention_mask`` (shape ``(B, prompt_len)``, 1=real, 0=pad) is
    given, also masks out padding slots in the key dimension so queries don't
    attend to left-pad tokens. The tokenizer runs with ``padding_side="left"``,
    so those slots are at the front of the sequence.
    """
    mask = torch.full(
        (batch_size, 1, prompt_len, max_cache_len), float("-inf"), dtype=dtype
    )
    for q_pos in range(prompt_len):
        mask[:, :, q_pos, : q_pos + 1] = 0.0
    if token_attention_mask is not None:
        tam = token_attention_mask.to(dtype=dtype)  # (B, prompt_len)
        pad_k = torch.where(
            tam > 0,
            torch.zeros((), dtype=dtype),
            torch.full((), float("-inf"), dtype=dtype),
        )  # 0 at real tokens, -inf at pad
        mask[:, :, :, :prompt_len] = (
            mask[:, :, :, :prompt_len] + pad_k[:, None, None, :]
        )
    return mask


def _build_decode_attention_mask(
    batch_size,
    current_pos_inclusive,
    max_cache_len,
    token_attention_mask=None,
    prompt_len=None,
    dtype=torch.bfloat16,
):
    """(B, 1, 1, max_cache_len) additive mask for a single decode step.

    If ``token_attention_mask`` + ``prompt_len`` are given, also masks cache
    slots ``0..prompt_len-1`` where the prompt was padding, so the decode
    step doesn't attend to left-pad KV entries that prefill wrote in.
    """
    mask = torch.full((batch_size, 1, 1, max_cache_len), float("-inf"), dtype=dtype)
    mask[:, :, :, :current_pos_inclusive] = 0.0
    if token_attention_mask is not None and prompt_len is not None:
        tam = token_attention_mask.to(dtype=dtype)  # (B, prompt_len)
        pad_k = torch.where(
            tam > 0,
            torch.zeros((), dtype=dtype),
            torch.full((), float("-inf"), dtype=dtype),
        )
        mask[:, :, :, :prompt_len] = (
            mask[:, :, :, :prompt_len] + pad_k[:, None, None, :]
        )
    return mask


def _apply_shard_specs(model, mesh, args, batch_size, tokens_device):
    """Apply mark_sharding to all weights + the input tokens."""
    xs.mark_sharding(tokens_device, mesh, ("_axis_1", None))
    xs.mark_sharding(model.embed.weight, mesh, (None, "_axis_0"))
    for layer in model.layers:
        attn = layer.attn
        xs.mark_sharding(attn.wq_b.weight, mesh, ("_axis_0", None))
        xs.mark_sharding(attn.wkv_b.weight, mesh, ("_axis_0", None))
        xs.mark_sharding(attn.wo.weight, mesh, (None, "_axis_0"))
        xs.mark_sharding(attn.wq_a.weight, mesh, (None, "_axis_0"))
        xs.mark_sharding(attn.wkv_a.weight, mesh, (None, "_axis_0"))
        cache_layer = model.past_key_values.layers[layer.layer_id]
        xs.mark_sharding(cache_layer.compressed_kv, mesh, ("_axis_1", None, None, None))
        xs.mark_sharding(cache_layer.k_pe, mesh, ("_axis_1", None, None, None))
        ffn = layer.ffn
        if hasattr(ffn, "mlp"):
            mlp = ffn.mlp
            xs.mark_sharding(mlp.router.gate.weight, mesh, (None, "_axis_0"))
            xs.mark_sharding(
                mlp.experts.gate_proj, mesh, (("_axis_0", "_axis_1"), None, None)
            )
            xs.mark_sharding(
                mlp.experts.up_proj, mesh, (("_axis_0", "_axis_1"), None, None)
            )
            xs.mark_sharding(
                mlp.experts.down_proj, mesh, (("_axis_0", "_axis_1"), None, None)
            )
            # Biases are None for DeepSeek v3.1 (no-bias MoE) — skip.
            if mlp.experts.gate_proj_bias is not None:
                xs.mark_sharding(
                    mlp.experts.gate_proj_bias,
                    mesh,
                    (("_axis_0", "_axis_1"), None),
                )
            if mlp.experts.up_proj_bias is not None:
                xs.mark_sharding(
                    mlp.experts.up_proj_bias, mesh, (("_axis_0", "_axis_1"), None)
                )
            if mlp.experts.down_proj_bias is not None:
                xs.mark_sharding(
                    mlp.experts.down_proj_bias, mesh, (("_axis_0", "_axis_1"), None)
                )
            shared = getattr(ffn, "shared_experts", None)
            if shared is not None:
                xs.mark_sharding(shared.w1.weight, mesh, (None, "_axis_0"))
                xs.mark_sharding(shared.w3.weight, mesh, (None, "_axis_0"))
                xs.mark_sharding(shared.w2.weight, mesh, ("_axis_0", None))
        else:
            xs.mark_sharding(ffn.w1.weight, mesh, ("_axis_1", "_axis_0"))
            xs.mark_sharding(ffn.w3.weight, mesh, ("_axis_1", "_axis_0"))
            xs.mark_sharding(ffn.w2.weight, mesh, ("_axis_0", "_axis_1"))
        xs.mark_sharding(layer.attn_norm.weight, mesh, ("_axis_0",))
        xs.mark_sharding(layer.ffn_norm.weight, mesh, ("_axis_0",))
    xs.mark_sharding(model.norm.weight, mesh, ("_axis_0",))
    xs.mark_sharding(model.head.weight, mesh, (None, "_axis_0"))


def _build_and_load_model(args, repo_id, prefer_cpu_capable=False):
    """Build a meta-device ModifiedTransformer and populate real weights.

    Two weight-load paths with different trade-offs. The caller picks via
    ``prefer_cpu_capable``; whether to actually *run* a CPU reference is a
    separate concern gated at the test level.

    - Post-sparse cache (``prefer_cpu_capable=False``, default — matches the
      old-branch speed profile): loads pre-stacked expert weights via mmap,
      then nulls ``_original_mlp`` + ``original_experts`` to free memory.
      ``enable_sparse_mlp`` runs on meta, so no runtime expert stacking.
      CPU forward through the sparse layer will fail — do not call
      ``model(tokens_cpu, ...)`` after this path.
    - Dequantized cache (``prefer_cpu_capable=True``): loads the per-expert
      BF16 cache, then ``enable_sparse_mlp`` runtime-stacks experts while
      keeping the original MoE module alive via ``cpu_forward_module`` —
      CPU forward works. For large n_layers this is *much* slower (both
      the stacking step and the subsequent move-to-device).

    If the preferred path's cache is missing, falls back to the other
    (logging). If neither cache exists, dequantizes from FP8 shards.

    Returns ``(model, mesh_shape)``.
    """
    with torch.device("meta"):
        model = ModifiedTransformer(args)
    mesh_shape = (4, 8)
    post_sparse_dir = _post_sparse_cache_dir(repo_id, args.n_layers)
    dequant_dir = _dequant_cache_dir(repo_id, args.n_layers)

    def _load_post_sparse():
        if not _has_cache(post_sparse_dir):
            print(
                f"[weights] post-sparse cache not found at {post_sparse_dir}, "
                "building... (self-heals pre-sparse if absent)",
                flush=True,
            )
            build_post_sparse_cache(repo_id, args.n_layers, args.n_dense_layers)
        print(f"[weights] loading post-sparse cache from {post_sparse_dir}", flush=True)
        enable_sparse_mlp(model, mesh=mesh_shape, cluster_axis=0, config=args)
        for _, mod in model.named_modules():
            if isinstance(mod, A2aSparseMLP):
                object.__setattr__(mod, "_original_mlp", None)
            if hasattr(mod, "original_experts"):
                mod.original_experts = nn.ModuleList()
        state_dict = _load_cache_chunked(post_sparse_dir)
        model.load_state_dict(state_dict, strict=False, assign=True)
        _fix_meta_buffers(model, args)
        model.eval()

    def _load_dequant():
        print(f"[weights] loading dequantized cache from {dequant_dir}", flush=True)
        _load_modified_dequantized_weights(
            model,
            repo_id=repo_id,
            n_layers=args.n_layers,
            n_dense_layers=args.n_dense_layers,
        )
        _fix_meta_buffers(model, args)
        model.eval()
        enable_sparse_mlp(model, mesh=mesh_shape, cluster_axis=0, config=args)

    if prefer_cpu_capable:
        if _has_cache(post_sparse_dir) and not _has_cache(dequant_dir):
            print(
                "[weights] prefer_cpu_capable=True but dequant cache missing; "
                "falling back to post-sparse cache (CPU forward will be broken)",
                flush=True,
            )
            _load_post_sparse()
        else:
            _load_dequant()
    else:
        if _has_cache(dequant_dir) and not _has_cache(post_sparse_dir):
            print(
                "[weights] post-sparse cache missing; falling back to "
                "dequant cache (slower — runtime expert stacking)",
                flush=True,
            )
            _load_dequant()
        else:
            _load_post_sparse()

    return model, mesh_shape


def _reset_caches(model):
    for cache_layer in model.past_key_values.layers:
        cache_layer.compressed_kv.zero_()
        cache_layer.k_pe.zero_()
    for layer in model.layers:
        if layer.attn.indexer is not None:
            layer.attn.indexer.k_cache.zero_()


def _pcc(a, b):
    x = a.to(torch.float64).flatten().numpy()
    y = b.to(torch.float64).flatten().numpy()
    vx = x - x.mean()
    vy = y - y.mean()
    denom = float(np.linalg.norm(vx) * np.linalg.norm(vy))
    return float("nan") if denom == 0 else float(np.dot(vx, vy) / denom)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_deepseek_v3_1_full_sparse_moe():
    """Prefill-only forward on the (4,8) mesh with A2aSparseMLP. Real input,
    real weights, CPU PCC comparison."""
    t_start = time.perf_counter()

    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    from transformers import AutoTokenizer

    batch_size = 32
    seq_len = 32

    repo_id = DEEPSEEK_V3_1_REPO
    args = load_deepseek_config(repo_id)
    args.n_layers = int(os.environ.get("DEEPSEEK_N_LAYERS", args.n_dense_layers + 1))
    args.max_batch_size = batch_size
    # max_seq_len must exceed original_seq_len (4096) to activate YaRN mscale.
    args.max_seq_len = args.original_seq_len + 1
    print(
        f"[config] n_layers={args.n_layers}, max_seq_len={args.max_seq_len}", flush=True
    )

    t0 = time.perf_counter()
    print(f"[timing] config + init: {t0 - t_start:.1f}s", flush=True)

    # DEEPSEEK_CPU_REFERENCE=1 runs the CPU eager reference and reports PCC.
    # Requires the dequant cache (which keeps the original MoE alive for CPU
    # forward); post-sparse cache is incompatible. Disabled by default so
    # large-layer runs use the fast post-sparse path.
    run_cpu_reference = os.environ.get("DEEPSEEK_CPU_REFERENCE", "0") == "1"

    model, mesh_shape = _build_and_load_model(
        args, repo_id, prefer_cpu_capable=run_cpu_reference
    )

    t1 = time.perf_counter()
    print(f"[timing] model build + weight load: {t1 - t0:.1f}s", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        repo_id, trust_remote_code=True, add_bos_token=True, padding_side="left"
    )
    prompt_text = (
        "Tenstorrent is a company that builds AI accelerators. "
        "Their chips are designed to"
    )
    encoded = tokenizer(
        prompt_text,
        return_tensors="pt",
        max_length=seq_len,
        truncation=True,
        padding="max_length",
    )
    tokens_single = encoded["input_ids"][:, :seq_len]
    tokens_cpu = tokens_single.repeat(batch_size, 1)
    print(
        f"[input] prompt: {tokenizer.decode(tokens_single[0].tolist())!r}", flush=True
    )

    # ===== CPU reference (opt-in via DEEPSEEK_CPU_REFERENCE=1) =====
    cpu_logits = None
    if run_cpu_reference:
        print("[cpu] running prefill eagerly (int path)...", flush=True)
        t_cpu_start = time.perf_counter()
        with torch.no_grad():
            cpu_logits = model(tokens_cpu, start_pos=0)
        t_cpu_end = time.perf_counter()
        cpu_top5 = cpu_logits[0].topk(5, dim=-1).indices
        print(f"[timing] CPU prefill: {t_cpu_end - t_cpu_start:.1f}s", flush=True)
        print(
            f"[CPU top-5] {[tokenizer.decode([t]) for t in cpu_top5.tolist()]}",
            flush=True,
        )
        _reset_caches(model)
    else:
        print(
            "[cpu] SKIPPED — set DEEPSEEK_CPU_REFERENCE=1 to run CPU golden",
            flush=True,
        )

    # ===== TT path =====
    torch._dynamo.reset()

    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("_axis_0", "_axis_1"))
    device = torch_xla.device()

    torch_xla.set_custom_compile_options(
        CompilerConfig(experimental_weight_dtype="bfp_bf8").to_torch_compile_options()
    )
    compiled_model = torch.compile(model, backend="tt")

    t_mv_start = time.perf_counter()
    model = model.to(device)
    tokens_device = tokens_cpu.to(device)
    t_mv_end = time.perf_counter()
    print(f"[timing] move to device: {t_mv_end - t_mv_start:.1f}s", flush=True)

    _apply_shard_specs(model, mesh, args, batch_size, tokens_device)

    print("[tt] running prefill...", flush=True)
    t_tt_start = time.perf_counter()
    with torch.no_grad():
        tt_logits = compiled_model(tokens_device, start_pos=0)
    tt_logits_cpu = tt_logits.to("cpu").detach()
    t_tt_end = time.perf_counter()
    print(
        f"[timing] TT prefill (compile + run): {t_tt_end - t_tt_start:.1f}s", flush=True
    )

    tt_top5 = tt_logits_cpu[0].topk(5, dim=-1).indices
    print(
        f"[TT  top-5] {[tokenizer.decode([t]) for t in tt_top5.tolist()]}", flush=True
    )

    if cpu_logits is not None:
        pcc = _pcc(tt_logits_cpu, cpu_logits.detach())
        print(f"[pcc] logits PCC (TT vs CPU) = {pcc:.4f} (required: 0.99)", flush=True)
    else:
        print("[pcc] SKIPPED — no CPU reference", flush=True)


def test_deepseek_v3_1_decode_static_cache():
    """Autoregressive greedy decode on TT using the MLACache path.

    Two compiles: prefill (prompt_seq_len), decode (seq_len=1, reused).
    """
    import torch._dynamo  # noqa: F401

    t_start = time.perf_counter()

    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    from transformers import AutoTokenizer

    batch_size = 32
    prompt_seq_len = 32
    n_decode = 10

    repo_id = DEEPSEEK_V3_1_REPO
    args = load_deepseek_config(repo_id)
    args.n_layers = int(os.environ.get("DEEPSEEK_N_LAYERS", args.n_dense_layers + 1))
    args.max_batch_size = batch_size
    args.max_seq_len = args.original_seq_len + 1  # = 4097, activates YaRN
    print(
        f"[config] n_layers={args.n_layers}, max_seq_len={args.max_seq_len}", flush=True
    )

    t0 = time.perf_counter()
    print(f"[timing] config + init: {t0 - t_start:.1f}s", flush=True)

    # DEEPSEEK_CPU_REFERENCE=1 runs the CPU eager reference (prefill + decode
    # loop) and reports per-step PCC. Requires the dequant cache; post-sparse
    # cache is incompatible. Disabled by default so 61-layer runs use the fast
    # post-sparse path.
    run_cpu_reference = os.environ.get("DEEPSEEK_CPU_REFERENCE", "0") == "1"

    model, mesh_shape = _build_and_load_model(
        args, repo_id, prefer_cpu_capable=run_cpu_reference
    )

    t1 = time.perf_counter()
    print(f"[timing] model build + weight load: {t1 - t0:.1f}s", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        repo_id, trust_remote_code=True, add_bos_token=True, padding_side="left"
    )
    prompt_text = "Tell me a short story."
    encoded = tokenizer(
        prompt_text,
        return_tensors="pt",
        max_length=prompt_seq_len,
        truncation=True,
        padding="max_length",
    )
    tokens_single = encoded["input_ids"][:, :prompt_seq_len]
    prompt_len = tokens_single.shape[1]
    tokens_cpu = tokens_single.repeat(batch_size, 1)
    # Tokenizer attention_mask: 1 for real tokens, 0 for left-pad. Used below
    # to mask pad slots out of prefill/decode attention (pad=<eos>, so without
    # this queries silently attend to pad KV entries).
    token_attn_mask_cpu = encoded["attention_mask"][:, :prompt_seq_len].repeat(
        batch_size, 1
    )
    print(
        f"[prompt] {prompt_len} tokens x{batch_size}: "
        f"{tokenizer.decode(tokens_single[0])!r}",
        flush=True,
    )

    # ===== CPU eager reference (opt-in via DEEPSEEK_CPU_REFERENCE=1) =====
    cpu_logits_list = []
    cpu_token_ids = []
    if run_cpu_reference:
        print("[cpu] running prefill + decode eagerly (int path)...", flush=True)
        t_cpu_start = time.perf_counter()
        with torch.no_grad():
            logits = model(tokens_cpu, start_pos=0)
            cpu_logits_list.append(logits[0].detach().clone())
            next_id = int(logits[0].argmax(dim=-1).item())
            cpu_token_ids.append(next_id)
            for step in range(n_decode - 1):
                next_tokens = torch.full(
                    (batch_size, 1), next_id, dtype=tokens_cpu.dtype
                )
                logits = model(next_tokens, start_pos=prompt_len + step)
                cpu_logits_list.append(logits[0].detach().clone())
                next_id = int(logits[0].argmax(dim=-1).item())
                cpu_token_ids.append(next_id)
        t_cpu_end = time.perf_counter()
        print(
            f"[timing] CPU prefill+decode: {t_cpu_end - t_cpu_start:.1f}s",
            flush=True,
        )
        print(
            f"[CPU tokens] {[tokenizer.decode([t]) for t in cpu_token_ids]}",
            flush=True,
        )
        print(
            f"[CPU full]   "
            f"{tokenizer.decode(tokens_single[0].tolist() + cpu_token_ids)!r}",
            flush=True,
        )
        _reset_caches(model)
    else:
        print(
            "[cpu] SKIPPED — set DEEPSEEK_CPU_REFERENCE=1 to run CPU golden",
            flush=True,
        )

    # ===== TT path =====
    torch._dynamo.reset()

    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("_axis_0", "_axis_1"))
    device = torch_xla.device()

    torch_xla.set_custom_compile_options(
        CompilerConfig(experimental_weight_dtype="bfp_bf8").to_torch_compile_options()
    )
    compiled_model = torch.compile(model, backend="tt")

    t_mv_start = time.perf_counter()
    model = model.to(device)
    tokens_device = tokens_cpu.to(device)
    cache_position_prefill = torch.arange(prompt_len, dtype=torch.long).to(device)
    attn_mask_prefill = _build_prefill_attention_mask(
        batch_size,
        prompt_len,
        args.max_seq_len,
        token_attention_mask=token_attn_mask_cpu,
        dtype=torch.bfloat16,
    ).to(device)
    t_mv_end = time.perf_counter()
    print(f"[timing] move to device: {t_mv_end - t_mv_start:.1f}s", flush=True)

    _apply_shard_specs(model, mesh, args, batch_size, tokens_device)

    # ===== Prefill =====
    print("[tt] running prefill (compile #1)...", flush=True)
    t_pf_start = time.perf_counter()
    with torch.no_grad():
        tt_logits_prefill = compiled_model(
            tokens_device,
            cache_position=cache_position_prefill,
            attention_mask=attn_mask_prefill,
        )
    t_pf_end = time.perf_counter()
    print(
        f"[timing] TT prefill compile + run: {t_pf_end - t_pf_start:.1f}s",
        flush=True,
    )
    tt_logits_prefill_cpu = tt_logits_prefill.to("cpu").detach()
    tt_logits_list = [tt_logits_prefill_cpu[0].clone()]
    tt_token_ids = [int(tt_logits_prefill_cpu[0].argmax(dim=-1).item())]
    print(
        f"[tt] prefill top-1: {tokenizer.decode([tt_token_ids[0]])!r}",
        flush=True,
    )

    # ===== Decode loop =====
    for step in range(n_decode - 1):
        pos = prompt_len + step
        next_tokens = torch.full(
            (batch_size, 1), tt_token_ids[-1], dtype=tokens_cpu.dtype
        ).to(device)
        cache_position_decode = torch.tensor([pos], dtype=torch.long).to(device)
        attn_mask_decode = _build_decode_attention_mask(
            batch_size,
            pos + 1,
            args.max_seq_len,
            token_attention_mask=token_attn_mask_cpu,
            prompt_len=prompt_len,
            dtype=torch.bfloat16,
        ).to(device)
        t_dec_start = time.perf_counter()
        with torch.no_grad():
            tt_logits = compiled_model(
                next_tokens,
                cache_position=cache_position_decode,
                attention_mask=attn_mask_decode,
            )
        tt_logits_cpu = tt_logits.to("cpu").detach()
        t_dec_end = time.perf_counter()
        tt_logits_list.append(tt_logits_cpu[0].clone())
        tt_token_ids.append(int(tt_logits_cpu[0].argmax(dim=-1).item()))
        print(
            f"[tt] decode step {step + 1}/{n_decode - 1}: "
            f"time={t_dec_end - t_dec_start:.2f}s "
            f"tok={tokenizer.decode([tt_token_ids[-1]])!r}",
            flush=True,
        )

    print(
        f"[TT tokens]  {[tokenizer.decode([t]) for t in tt_token_ids]}",
        flush=True,
    )
    print(
        f"[TT full]    "
        f"{tokenizer.decode(tokens_single[0].tolist() + tt_token_ids)!r}",
        flush=True,
    )

    if cpu_logits_list:
        print("\n[pcc] per-step logits PCC (TT vs CPU):", flush=True)
        step_pccs = []
        for step, (tt_l, cpu_l) in enumerate(zip(tt_logits_list, cpu_logits_list)):
            pcc = _pcc(tt_l, cpu_l)
            step_pccs.append(pcc)
            tt_tok = tt_token_ids[step]
            cpu_tok = cpu_token_ids[step]
            match = "OK" if tt_tok == cpu_tok else "MISMATCH"
            print(
                f"  step {step}: pcc={pcc:.4f}  "
                f"tt_tok={tt_tok!r:>10} cpu_tok={cpu_tok!r:>10}  [{match}]",
                flush=True,
            )
        print(
            f"\n[pcc] min={min(step_pccs):.4f}, max={max(step_pccs):.4f}, "
            f"avg={sum(step_pccs) / len(step_pccs):.4f}",
            flush=True,
        )
        matches = sum(1 for a, b in zip(tt_token_ids, cpu_token_ids) if a == b)
        print(
            f"[tokens] TT==CPU match: {matches}/{len(tt_token_ids)}",
            flush=True,
        )
    else:
        print("\n[pcc] SKIPPED — no CPU reference", flush=True)
