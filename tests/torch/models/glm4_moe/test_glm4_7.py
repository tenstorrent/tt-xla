# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""GLM-4.7 full sparse-MoE prefill test on Galaxy (4,8) mesh.

Loads real BF16 weights via a post-sparse cache built by
``build_weight_cache_glm.py``. Runs a single prefill through
``Glm4MoeForCausalLM`` with ``A2aSparseMLP`` MoE layers, and prints the
top-k next-token predictions. No CPU reference.
"""
import json
import os
import re
import sys
import time

import numpy as np
import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from huggingface_hub import hf_hub_download
from infra.testers.compiler_config import CompilerConfig
from safetensors.torch import load_file as safetensors_load_file
from torch import nn
from torch_xla.distributed.spmd import Mesh, mark_sharding
from transformers import AutoTokenizer
from transformers.cache_utils import StaticCache
from transformers.models.glm4_moe.configuration_glm4_moe import Glm4MoeConfig
from transformers.models.glm4_moe.modeling_glm4_moe import (
    Glm4MoeForCausalLM,
    Glm4MoeRotaryEmbedding,
)
from tt_torch.sparse_mlp import A2aSparseMLP, build_expert_mapping, enable_sparse_mlp

# Import shard-streaming + cache helpers from the builder script (single source of truth)
sys.path.insert(0, os.path.dirname(__file__))
from build_weight_cache_glm import (
    _group_by_shard,
    _has_cache,
    _load_tensors,
    _post_sparse_cache_dir,
    build_post_sparse_cache,
)

GLM_REPO = "zai-org/GLM-4.7"


# TODO(issue): link tracking issue once filed.
# Stock Glm4MoeTopkRouter.forward (transformers modeling_glm4_moe.py) upcasts
# both input and weight to float32 before the gate linear. On TT, the fp32
# linear kernel has accuracy issues large enough to flip top-k routing
# decisions; wrong expert selections then compound nonlinearly across the
# MoE stack and produce degenerate output (e.g. " chip chip chip ..." at
# 92 layers). DeepSeek v3.1 hits the same issue and works around it in
# tests/torch/models/deepseek_v3_2_exp/modified_model.py:1080 with the
# comment: "fp32 linear has accuracy issues — keep in bf16 for Gate routing".
# We apply the same workaround here by keeping the router linear in bf16.
def _patch_router_bf16():
    import torch.nn.functional as F  # noqa: F401
    from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeTopkRouter

    def _bf16_forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        return F.linear(hidden_states, self.weight)

    Glm4MoeTopkRouter.forward = _bf16_forward


# ---------- helpers ----------
def _load_cache_chunked(cache_dir):
    state_dict = {}
    for fname in sorted(os.listdir(cache_dir)):
        if not fname.endswith(".safetensors"):
            continue
        state_dict.update(safetensors_load_file(os.path.join(cache_dir, fname)))
    return state_dict


def _restore_router_bias_fp32(model):
    """HF declares ``e_score_correction_bias`` in ``_keep_in_fp32_modules_strict``
    because bf16 truncation at its magnitude (~5) gives ~0.04 rounding error,
    which is enough to flip MoE routing decisions. Our cache stores all tensors
    as bf16 indiscriminately; cast this buffer back to fp32 after load."""
    from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeTopkRouter

    n_fixed = 0
    for module in model.modules():
        if isinstance(module, Glm4MoeTopkRouter):
            if module.e_score_correction_bias.dtype != torch.float32:
                module.e_score_correction_bias = module.e_score_correction_bias.to(
                    torch.float32
                )
                n_fixed += 1
    if n_fixed:
        print(
            f"[precision] restored e_score_correction_bias to fp32 in {n_fixed} routers",
            flush=True,
        )


def _fix_meta_expert_mapping(model):
    """Rebuild A2aSparseMLP.expert_mapping on CPU (meta path leaves it empty)."""
    for _, mod in model.named_modules():
        if isinstance(mod, A2aSparseMLP):
            mapping = build_expert_mapping(mod.num_experts, mod.num_devices)
            mod.expert_mapping = mapping


def _materialize_rope_buffer(model, config):
    """Re-initialize Glm4MoeRotaryEmbedding on CPU. Its inv_freq buffer is
    computed from config during __init__, so a meta-built model has meta
    inv_freq with no data. Replace the module with a fresh CPU instance."""
    model.model.rotary_emb = Glm4MoeRotaryEmbedding(config, device="cpu")


def _build_and_load_model_post_sparse(config, repo_id, mesh_shape):
    """Meta-init + enable_sparse_mlp + load post-sparse cache. Fast, TT-only."""
    cache_dir = _post_sparse_cache_dir(repo_id, config.num_hidden_layers)
    if not _has_cache(cache_dir):
        print(
            f"[weights] post-sparse cache not found at {cache_dir}, building...",
            flush=True,
        )
        build_post_sparse_cache(
            repo_id,
            config.num_hidden_layers,
            config.first_k_dense_replace,
            config.n_routed_experts,
        )

    print(
        f"[weights] meta-building Glm4MoeForCausalLM (n_layers={config.num_hidden_layers})",
        flush=True,
    )
    t0 = time.perf_counter()
    with torch.device("meta"):
        model = Glm4MoeForCausalLM(config)
    print(f"[timing] meta init: {time.perf_counter() - t0:.1f}s", flush=True)

    t0 = time.perf_counter()
    enable_sparse_mlp(model, mesh=mesh_shape, cluster_axis=0, config=config)
    # Free original expert references — cache has stacked form; no CPU forward.
    for _, mod in model.named_modules():
        if isinstance(mod, A2aSparseMLP):
            object.__setattr__(mod, "_original_mlp", None)
        if hasattr(mod, "original_experts"):
            mod.original_experts = nn.ModuleList()
    print(f"[timing] enable_sparse_mlp: {time.perf_counter() - t0:.1f}s", flush=True)

    t0 = time.perf_counter()
    print(f"[weights] loading post-sparse cache from {cache_dir}", flush=True)
    state_dict = _load_cache_chunked(cache_dir)
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    print(f"[timing] cache load: {time.perf_counter() - t0:.1f}s", flush=True)
    print(
        f"[weights] loaded {len(state_dict)} tensors; "
        f"missing={len(missing)}, unexpected={len(unexpected)}",
        flush=True,
    )
    if missing:
        print(f"[weights] missing (first 5): {missing[:5]}", flush=True)
    if unexpected:
        print(f"[weights] unexpected (first 5): {unexpected[:5]}", flush=True)

    _fix_meta_expert_mapping(model)
    _materialize_rope_buffer(model, config)
    _restore_router_bias_fp32(model)
    model.eval()
    return model


def _load_cpu_state_dict_shard_on_demand(config, repo_id):
    """Read only the HF shards needed for the given n_layers. Returns CPU state_dict.

    Applies the ``qwen2_moe`` per-expert -> fused conversion that transformers
    runs automatically in ``from_pretrained`` (GLM-4.7 is aliased to
    ``qwen2_moe`` in conversion_mapping.py). MoE layer weights on disk live at
    ``experts.{j}.{gate,up,down}_proj.weight`` but the ``Glm4MoeNaiveMoe``
    module expects fused ``experts.gate_up_proj [E, 2*inter, H]`` and
    ``experts.down_proj [E, H, inter]``.

    Filters out keys for layers beyond n_layers and any mtp/next-n keys.
    """
    index_path = hf_hub_download(repo_id, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    n_layers = config.num_hidden_layers
    n_dense = config.first_k_dense_replace
    n_experts = config.n_routed_experts

    expert_re = re.compile(
        r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
    )

    # Filter out keys for layers >= n_layers and any mtp/next-n keys.
    wanted_keys = []
    for ckpt_key in weight_map:
        m = re.match(r"model\.layers\.(\d+)\.", ckpt_key)
        if m and int(m.group(1)) >= n_layers:
            continue
        if ckpt_key.startswith("mtp.") or ".mtp." in ckpt_key:
            continue
        wanted_keys.append(ckpt_key)

    shard_to_keys = _group_by_shard(wanted_keys, weight_map)
    raw = _load_tensors(shard_to_keys, repo_id)

    # Split into per-expert (for fusion below) vs. pass-through.
    expert_tensors = {}  # (layer_idx, expert_idx, name) -> tensor
    state_dict = {}
    for k, t in raw.items():
        t = t.to(torch.bfloat16)
        m = expert_re.match(k)
        if m:
            layer_idx = int(m.group(1))
            if layer_idx < n_dense:
                raise RuntimeError(f"Unexpected per-expert key on dense layer: {k}")
            expert_tensors[(layer_idx, int(m.group(2)), m.group(3))] = t
        else:
            state_dict[k] = t
    del raw

    # Fuse per-expert -> gate_up_proj + down_proj per MoE layer.
    moe_layers = sorted({lk[0] for lk in expert_tensors.keys()})
    for layer_idx in moe_layers:
        gate_stack = torch.stack(
            [expert_tensors[(layer_idx, j, "gate_proj")] for j in range(n_experts)],
            dim=0,
        )  # [E, inter, H]
        up_stack = torch.stack(
            [expert_tensors[(layer_idx, j, "up_proj")] for j in range(n_experts)],
            dim=0,
        )  # [E, inter, H]
        down_stack = torch.stack(
            [expert_tensors[(layer_idx, j, "down_proj")] for j in range(n_experts)],
            dim=0,
        )  # [E, H, inter]
        # Concatenate gate + up on dim 1 (out-features axis).
        gate_up = torch.cat(
            [gate_stack, up_stack], dim=1
        ).contiguous()  # [E, 2*inter, H]
        del gate_stack, up_stack

        state_dict[f"model.layers.{layer_idx}.mlp.experts.gate_up_proj"] = gate_up
        state_dict[f"model.layers.{layer_idx}.mlp.experts.down_proj"] = (
            down_stack.contiguous()
        )

        # Drop source tensors for this layer to keep peak memory in check.
        for j in range(n_experts):
            for name in ("gate_proj", "up_proj", "down_proj"):
                expert_tensors.pop((layer_idx, j, name), None)

    return state_dict


def _build_and_load_model_for_cpu(config, repo_id, mesh_shape):
    """Meta-init + shard-on-demand HF load + enable_sparse_mlp with originals alive."""
    print(
        f"[weights] shard-on-demand HF load (n_layers={config.num_hidden_layers})",
        flush=True,
    )
    t0 = time.perf_counter()
    with torch.device("meta"):
        model = Glm4MoeForCausalLM(config)
    print(f"[timing] meta init: {time.perf_counter() - t0:.1f}s", flush=True)

    t0 = time.perf_counter()
    state_dict = _load_cpu_state_dict_shard_on_demand(config, repo_id)
    print(
        f"[timing] HF shard load: {time.perf_counter() - t0:.1f}s "
        f"({len(state_dict)} tensors)",
        flush=True,
    )

    t0 = time.perf_counter()
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    # Free the state_dict dict now that params reference the tensors directly.
    del state_dict
    print(
        f"[timing] load_state_dict: {time.perf_counter() - t0:.1f}s "
        f"(missing={len(missing)}, unexpected={len(unexpected)})",
        flush=True,
    )
    if missing:
        print(f"[weights] missing (first 5): {missing[:5]}", flush=True)
    if unexpected:
        print(f"[weights] unexpected (first 5): {unexpected[:5]}", flush=True)

    _materialize_rope_buffer(model, config)
    _restore_router_bias_fp32(model)

    t0 = time.perf_counter()
    enable_sparse_mlp(model, mesh=mesh_shape, cluster_axis=0, config=config)
    # enable_sparse_mlp stacks experts: creates NEW stacked tensors AND keeps
    # original per-expert tensors alive via _original_mlp / original_experts so
    # CPU forward works. Peak memory doubles transiently.
    print(f"[timing] enable_sparse_mlp: {time.perf_counter() - t0:.1f}s", flush=True)

    model.eval()
    return model


def _release_cpu_originals(model):
    """After CPU forward, drop the per-expert original tensors so they don't
    get transferred to TT. A2aSparseMLP keeps them alive via
    ``_original_mlp`` and ``original_experts``."""
    for _, mod in model.named_modules():
        if isinstance(mod, A2aSparseMLP):
            object.__setattr__(mod, "_original_mlp", None)
        if hasattr(mod, "original_experts"):
            mod.original_experts = nn.ModuleList()
    import gc

    gc.collect()


def _build_and_load_model(config, repo_id, mesh_shape, prefer_cpu_capable=False):
    if prefer_cpu_capable:
        return _build_and_load_model_for_cpu(config, repo_id, mesh_shape)
    return _build_and_load_model_post_sparse(config, repo_id, mesh_shape)


def _pcc(a, b):
    x = a.to(torch.float64).flatten().numpy()
    y = b.to(torch.float64).flatten().numpy()
    vx = x - x.mean()
    vy = y - y.mean()
    denom = float(np.linalg.norm(vx) * np.linalg.norm(vy))
    return float("nan") if denom == 0 else float(np.dot(vx, vy) / denom)


def _build_full_attention_mask(token_attn_mask, max_cache_len, batch_size):
    """2D attention mask of shape (B, max_cache_len), matching examples/pytorch/llama.py.

    ``token_attn_mask`` is the tokenizer's [B, prompt_len] mask (1=real, 0=pad).
    The returned mask has the prompt's mask in the first prompt_len slots and
    1s in the remaining slots (future decode positions are valid). HF's
    ``create_causal_mask`` converts this to a proper 4D causal+pad-aware
    mask internally — no NaN risk. The same mask object is reused across
    prefill and every decode step (no per-step rebuild).

    Sized to ``max_cache_len`` to prevent transformers from implicitly padding
    or recompiling, per the comment in the Llama example.
    """
    prompt_len = token_attn_mask.shape[1]
    full = torch.ones((batch_size, max_cache_len), dtype=token_attn_mask.dtype)
    full[:, :prompt_len] = token_attn_mask
    return full


def _apply_shard_specs(
    model, mesh, input_ids, position_ids, cache_position, static_cache, config
):
    """Mark sharding on inputs, kv cache, and every parameter."""
    num_dense = config.first_k_dense_replace

    # Inputs
    mark_sharding(input_ids, mesh, ("_axis_1", None))
    mark_sharding(position_ids, mesh, (None, None))
    mark_sharding(cache_position, mesh, (None,))

    # KV cache per layer
    for layer_cache in static_cache.layers:
        mark_sharding(layer_cache.keys, mesh, ("_axis_1", None, None, None))
        mark_sharding(layer_cache.values, mesh, ("_axis_1", None, None, None))

    base = model.model  # Glm4MoeModel
    mark_sharding(base.embed_tokens.weight, mesh, (None, "_axis_0"))
    mark_sharding(base.norm.weight, mesh, ("_axis_0",))
    mark_sharding(model.lm_head.weight, mesh, (None, "_axis_0"))

    for i, decoder_layer in enumerate(base.layers):
        attn = decoder_layer.self_attn
        mark_sharding(attn.q_proj.weight, mesh, ("_axis_0", None))
        mark_sharding(attn.k_proj.weight, mesh, ("_axis_0", None))
        mark_sharding(attn.v_proj.weight, mesh, ("_axis_0", None))
        mark_sharding(attn.o_proj.weight, mesh, (None, "_axis_0"))
        if attn.q_proj.bias is not None:
            mark_sharding(attn.q_proj.bias, mesh, ("_axis_0",))
            mark_sharding(attn.k_proj.bias, mesh, ("_axis_0",))
            mark_sharding(attn.v_proj.bias, mesh, ("_axis_0",))
        # GLM-4.7 has use_qk_norm=True
        if hasattr(attn, "q_norm"):
            mark_sharding(attn.q_norm.weight, mesh, ("_axis_0",))
            mark_sharding(attn.k_norm.weight, mesh, ("_axis_0",))

        mark_sharding(decoder_layer.input_layernorm.weight, mesh, ("_axis_0",))
        mark_sharding(decoder_layer.post_attention_layernorm.weight, mesh, ("_axis_0",))

        mlp_wrapper = decoder_layer.mlp
        if i < num_dense:
            # Dense Glm4MoeMLP: weight is [out, in] in nn.Linear convention
            mark_sharding(mlp_wrapper.gate_proj.weight, mesh, ("_axis_1", "_axis_0"))
            mark_sharding(mlp_wrapper.up_proj.weight, mesh, ("_axis_1", "_axis_0"))
            mark_sharding(mlp_wrapper.down_proj.weight, mesh, ("_axis_0", "_axis_1"))
        else:
            inner = mlp_wrapper.mlp  # A2aSparseMLP
            mark_sharding(inner.router.gate.weight, mesh, (None, "_axis_0"))
            mark_sharding(
                inner.experts.gate_proj,
                mesh,
                (("_axis_0", "_axis_1"), None, None),
            )
            mark_sharding(
                inner.experts.up_proj,
                mesh,
                (("_axis_0", "_axis_1"), None, None),
            )
            mark_sharding(
                inner.experts.down_proj,
                mesh,
                (("_axis_0", "_axis_1"), None, None),
            )
            if inner.experts.gate_proj_bias is not None:
                mark_sharding(
                    inner.experts.gate_proj_bias,
                    mesh,
                    (("_axis_0", "_axis_1"), None),
                )
            if inner.experts.up_proj_bias is not None:
                mark_sharding(
                    inner.experts.up_proj_bias,
                    mesh,
                    (("_axis_0", "_axis_1"), None),
                )
            if inner.experts.down_proj_bias is not None:
                mark_sharding(
                    inner.experts.down_proj_bias,
                    mesh,
                    (("_axis_0", "_axis_1"), None),
                )
            shared = getattr(mlp_wrapper, "shared_experts", None)
            if shared is not None:
                mark_sharding(shared.gate_proj.weight, mesh, (None, "_axis_0"))
                mark_sharding(shared.up_proj.weight, mesh, (None, "_axis_0"))
                mark_sharding(shared.down_proj.weight, mesh, ("_axis_0", None))


# ---------- test ----------
@pytest.mark.llmbox
@pytest.mark.nightly
def test_glm4_7_full_sparse_moe():
    t_start = time.perf_counter()

    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()
    _patch_router_bf16()

    batch_size = 32
    seq_len = 32
    n_layers = int(os.environ.get("GLM_N_LAYERS", 4))
    mesh_shape = (4, 8)
    run_cpu_reference = os.environ.get("GLM_CPU_REFERENCE", "0") == "1"

    # ----- Config -----
    config = Glm4MoeConfig.from_pretrained(GLM_REPO)
    config.num_hidden_layers = n_layers
    config.use_cache = True
    config._attn_implementation = "eager"
    # Don't build MTP/next-n head — not needed for prefill.
    if hasattr(config, "num_nextn_predict_layers"):
        config.num_nextn_predict_layers = 0
    print(
        f"[config] n_layers={n_layers}, n_dense={config.first_k_dense_replace}, "
        f"n_experts={config.n_routed_experts}, top_k={config.num_experts_per_tok}, "
        f"cpu_reference={run_cpu_reference}",
        flush=True,
    )

    # ----- Model + weights -----
    model = _build_and_load_model(
        config, GLM_REPO, mesh_shape, prefer_cpu_capable=run_cpu_reference
    )

    # ----- Tokenizer / inputs -----
    tokenizer = AutoTokenizer.from_pretrained(
        GLM_REPO, trust_remote_code=True, padding_side="left"
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
    token_attn_mask_cpu = encoded["attention_mask"][:, :seq_len].repeat(batch_size, 1)
    print(
        f"[input] prompt: {tokenizer.decode(tokens_single[0].tolist())!r}", flush=True
    )
    print(
        f"[input] real tokens: {int(token_attn_mask_cpu[0].sum())}/{seq_len}",
        flush=True,
    )

    # HF default: position_ids = cache_position.unsqueeze(0). RoPE is purely
    # relative, so absolute offset doesn't affect attention scores.
    cache_position = torch.arange(seq_len)
    position_ids = cache_position.unsqueeze(0).expand(batch_size, -1).contiguous()

    # StaticCache — sized exactly to the prefill (no decode in this test).
    max_cache_len = seq_len

    def _make_static_cache():
        cache = StaticCache(
            config=config,
            max_batch_size=batch_size,
            max_cache_len=max_cache_len,
            device="cpu",
            dtype=torch.bfloat16,
        )
        cache.early_initialization(
            batch_size=batch_size,
            num_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            dtype=torch.bfloat16,
            device="cpu",
        )
        return cache

    full_attn_mask = _build_full_attention_mask(
        token_attn_mask_cpu, max_cache_len, batch_size
    )

    # ===== CPU reference (opt-in via GLM_CPU_REFERENCE=1) =====
    cpu_logits = None
    if run_cpu_reference:
        print("[cpu] running prefill eagerly...", flush=True)
        t_cpu_start = time.perf_counter()
        cpu_cache = _make_static_cache()
        with torch.no_grad():
            cpu_out = model(
                input_ids=tokens_cpu,
                attention_mask=full_attn_mask,
                position_ids=position_ids,
                past_key_values=cpu_cache,
                cache_position=cache_position,
                use_cache=True,
            )
        # Detach + clone so cpu_logits doesn't keep cpu_out's graph alive.
        cpu_logits = cpu_out.logits.detach().clone()
        del cpu_out, cpu_cache
        print(
            f"[timing] CPU prefill: {time.perf_counter() - t_cpu_start:.1f}s",
            flush=True,
        )
        cpu_top5 = cpu_logits[0, -1].topk(5)
        cpu_tokens = [tokenizer.decode([t]) for t in cpu_top5.indices.tolist()]
        print(f"[CPU top-5] tokens={cpu_tokens}", flush=True)
        print(f"[CPU top-5] logits={cpu_top5.values.tolist()}", flush=True)

        # Drop the per-expert originals so only stacked weights transfer to TT.
        _release_cpu_originals(model)
    else:
        print("[cpu] SKIPPED — set GLM_CPU_REFERENCE=1 to run CPU golden", flush=True)

    # ----- TT path -----
    torch._dynamo.reset()

    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("_axis_0", "_axis_1"))
    device = torch_xla.device()

    torch_xla.set_custom_compile_options(
        CompilerConfig(experimental_weight_dtype="bfp_bf8").to_torch_compile_options()
    )

    # Move to device — use to_empty + load_state_dict for meta path; to() for CPU path.
    t0 = time.perf_counter()
    model = model.to(device)
    tokens_device = tokens_cpu.to(device)
    position_ids_device = position_ids.to(device)
    cache_position_device = cache_position.to(device)
    full_attn_mask_device = full_attn_mask.to(device)
    tt_cache = _make_static_cache()
    for layer_cache in tt_cache.layers:
        layer_cache.keys = layer_cache.keys.to(device)
        layer_cache.values = layer_cache.values.to(device)
    print(f"[timing] move to device: {time.perf_counter() - t0:.1f}s", flush=True)

    _apply_shard_specs(
        model,
        mesh,
        tokens_device,
        position_ids_device,
        cache_position_device,
        tt_cache,
        config,
    )
    mark_sharding(full_attn_mask_device, mesh, ("_axis_1", None))

    compiled_model = torch.compile(model, backend="tt")

    print("[tt] running prefill (compile + run)...", flush=True)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = compiled_model(
            input_ids=tokens_device,
            attention_mask=full_attn_mask_device,
            position_ids=position_ids_device,
            past_key_values=tt_cache,
            cache_position=cache_position_device,
            use_cache=True,
        )
    logits_tt_cpu = out.logits.to("cpu").detach()
    print(f"[timing] TT prefill: {time.perf_counter() - t0:.1f}s", flush=True)

    # Top-5 next-token predictions from the last prefill position.
    top5 = logits_tt_cpu[0, -1].topk(5)
    tokens = [tokenizer.decode([t]) for t in top5.indices.tolist()]
    print(f"[TT top-5] tokens={tokens}", flush=True)
    print(f"[TT top-5] logits={top5.values.tolist()}", flush=True)

    if cpu_logits is not None:
        pcc = _pcc(logits_tt_cpu, cpu_logits)
        print(f"[pcc] logits PCC (TT vs CPU) = {pcc:.4f} (required: 0.99)", flush=True)
        assert pcc >= 0.99, f"PCC {pcc:.4f} below required 0.99"

    print(f"[timing] total: {time.perf_counter() - t_start:.1f}s", flush=True)


# ---------- decode test ----------
@pytest.mark.llmbox
@pytest.mark.nightly
def test_glm4_7_decode_static_cache():
    """Autoregressive greedy decode on TT using StaticCache.

    Two torch.compile traces: prefill (seq=prompt_len) and decode (seq=1).
    With GLM_CPU_REFERENCE=1, also runs CPU eager reference and compares
    per-step logits PCC.
    """
    t_start = time.perf_counter()

    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()
    _patch_router_bf16()

    batch_size = 32
    prompt_seq_len = 32
    n_decode = 10
    n_layers = int(os.environ.get("GLM_N_LAYERS", 4))
    mesh_shape = (4, 8)
    run_cpu_reference = os.environ.get("GLM_CPU_REFERENCE", "0") == "1"

    # ----- Config -----
    config = Glm4MoeConfig.from_pretrained(GLM_REPO)
    config.num_hidden_layers = n_layers
    config.use_cache = True
    config._attn_implementation = "eager"
    if hasattr(config, "num_nextn_predict_layers"):
        config.num_nextn_predict_layers = 0
    print(
        f"[config] n_layers={n_layers}, n_dense={config.first_k_dense_replace}, "
        f"n_experts={config.n_routed_experts}, cpu_reference={run_cpu_reference}",
        flush=True,
    )

    # ----- Model + weights -----
    model = _build_and_load_model(
        config, GLM_REPO, mesh_shape, prefer_cpu_capable=run_cpu_reference
    )

    # ----- Tokenizer / inputs -----
    tokenizer = AutoTokenizer.from_pretrained(
        GLM_REPO, trust_remote_code=True, padding_side="left"
    )
    prompt_text = (
        "Tenstorrent is a company that builds AI accelerators. "
        "Their chips are designed to"
    )
    encoded = tokenizer(
        prompt_text,
        return_tensors="pt",
        max_length=prompt_seq_len,
        truncation=True,
        padding="max_length",
    )
    tokens_single = encoded["input_ids"][:, :prompt_seq_len]
    attn_single = encoded["attention_mask"][:, :prompt_seq_len]
    tokens_cpu = tokens_single.repeat(batch_size, 1)
    token_attn_mask_cpu = attn_single.repeat(batch_size, 1)
    prompt_real_len = int(token_attn_mask_cpu[0].sum())
    print(
        f"[input] prompt: {tokenizer.decode(tokens_single[0].tolist())!r}", flush=True
    )
    print(
        f"[input] real tokens: {prompt_real_len}/{prompt_seq_len}",
        flush=True,
    )

    max_cache_len = prompt_seq_len + n_decode

    def _make_static_cache():
        cache = StaticCache(
            config=config,
            max_batch_size=batch_size,
            max_cache_len=max_cache_len,
            device="cpu",
            dtype=torch.bfloat16,
        )
        cache.early_initialization(
            batch_size=batch_size,
            num_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            dtype=torch.bfloat16,
            device="cpu",
        )
        return cache

    # HF default: position_ids = cache_position.unsqueeze(0). RoPE is purely
    # relative, so absolute offset doesn't affect attention scores.
    prefill_cache_position = torch.arange(prompt_seq_len)
    prefill_position_ids = (
        prefill_cache_position.unsqueeze(0).expand(batch_size, -1).contiguous()
    )
    # Single 2D attention_mask sized to max_cache_len, reused across prefill
    # and every decode step. HF's create_causal_mask handles the conversion.
    full_attn_mask = _build_full_attention_mask(
        token_attn_mask_cpu, max_cache_len, batch_size
    )

    # ===== CPU reference =====
    cpu_logits_list = []
    cpu_token_ids = []
    if run_cpu_reference:
        print("[cpu] running prefill + decode eagerly...", flush=True)
        t_cpu_start = time.perf_counter()
        cpu_cache = _make_static_cache()
        with torch.no_grad():
            # Prefill
            cpu_out = model(
                input_ids=tokens_cpu,
                attention_mask=full_attn_mask,
                position_ids=prefill_position_ids,
                past_key_values=cpu_cache,
                cache_position=prefill_cache_position,
                use_cache=True,
            )
            # Keep last-position logits only (batch 0).
            last_logits = cpu_out.logits[:, -1].detach().clone()
            cpu_logits_list.append(last_logits[0])
            next_id = int(last_logits[0].argmax(dim=-1).item())
            cpu_token_ids.append(next_id)
            del cpu_out, last_logits

            # Decode — real token positions continue from prompt_real_len.
            # Same full_attn_mask is reused; only input_ids/position_ids/cache_position change.
            for step in range(n_decode - 1):
                cache_slot = prompt_seq_len + step
                real_pos = cache_slot
                next_tokens = torch.full(
                    (batch_size, 1), next_id, dtype=tokens_cpu.dtype
                )
                pos_ids = torch.tensor([[real_pos]], dtype=torch.long).repeat(
                    batch_size, 1
                )
                cache_pos = torch.tensor([cache_slot], dtype=torch.long)
                cpu_out = model(
                    input_ids=next_tokens,
                    attention_mask=full_attn_mask,
                    position_ids=pos_ids,
                    past_key_values=cpu_cache,
                    cache_position=cache_pos,
                    use_cache=True,
                )
                last_logits = cpu_out.logits[:, -1].detach().clone()
                cpu_logits_list.append(last_logits[0])
                next_id = int(last_logits[0].argmax(dim=-1).item())
                cpu_token_ids.append(next_id)
                del cpu_out, last_logits
        del cpu_cache
        print(
            f"[timing] CPU prefill+decode: {time.perf_counter() - t_cpu_start:.1f}s",
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
        # Drop CPU originals before TT path.
        _release_cpu_originals(model)
    else:
        print("[cpu] SKIPPED — set GLM_CPU_REFERENCE=1 to run CPU golden", flush=True)

    # ===== TT path =====
    torch._dynamo.reset()

    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("_axis_0", "_axis_1"))
    device = torch_xla.device()

    torch_xla.set_custom_compile_options(
        CompilerConfig(experimental_weight_dtype="bfp_bf8").to_torch_compile_options()
    )

    # Move model + tokens + fresh KV cache to device
    t0 = time.perf_counter()
    model = model.to(device)
    tokens_device = tokens_cpu.to(device)
    prefill_position_ids_device = prefill_position_ids.to(device)
    prefill_cache_position_device = prefill_cache_position.to(device)
    full_attn_mask_device = full_attn_mask.to(device)
    tt_cache = _make_static_cache()
    for layer_cache in tt_cache.layers:
        layer_cache.keys = layer_cache.keys.to(device)
        layer_cache.values = layer_cache.values.to(device)
    print(f"[timing] move to device: {time.perf_counter() - t0:.1f}s", flush=True)

    _apply_shard_specs(
        model,
        mesh,
        tokens_device,
        prefill_position_ids_device,
        prefill_cache_position_device,
        tt_cache,
        config,
    )
    mark_sharding(full_attn_mask_device, mesh, ("_axis_1", None))

    compiled_model = torch.compile(model, backend="tt")

    tt_logits_list = []
    tt_token_ids = []

    # ----- TT prefill -----
    print("[tt] running prefill (compile #1)...", flush=True)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = compiled_model(
            input_ids=tokens_device,
            attention_mask=full_attn_mask_device,
            position_ids=prefill_position_ids_device,
            past_key_values=tt_cache,
            cache_position=prefill_cache_position_device,
            use_cache=True,
        )
    prefill_last = out.logits[:, -1].to("cpu").detach()
    print(f"[timing] TT prefill: {time.perf_counter() - t0:.1f}s", flush=True)
    tt_logits_list.append(prefill_last[0])
    next_id = int(prefill_last[0].argmax(dim=-1).item())
    tt_token_ids.append(next_id)
    print(f"[tt] prefill top-1: {tokenizer.decode([next_id])!r}", flush=True)

    # ----- TT decode loop -----
    # Same full_attn_mask_device is reused (shape stable, no per-step rebuild).
    for step in range(n_decode - 1):
        cache_slot = prompt_seq_len + step
        real_pos = cache_slot
        next_tokens_cpu = torch.full((batch_size, 1), next_id, dtype=tokens_cpu.dtype)
        next_tokens_device = next_tokens_cpu.to(device)
        pos_ids_device = (
            torch.tensor([[real_pos]], dtype=torch.long)
            .repeat(batch_size, 1)
            .to(device)
        )
        cache_pos_device = torch.tensor([cache_slot], dtype=torch.long).to(device)

        mark_sharding(next_tokens_device, mesh, ("_axis_1", None))
        mark_sharding(pos_ids_device, mesh, (None, None))
        mark_sharding(cache_pos_device, mesh, (None,))

        t_step = time.perf_counter()
        with torch.no_grad():
            out = compiled_model(
                input_ids=next_tokens_device,
                attention_mask=full_attn_mask_device,
                position_ids=pos_ids_device,
                past_key_values=tt_cache,
                cache_position=cache_pos_device,
                use_cache=True,
            )
        step_last = out.logits[:, -1].to("cpu").detach()
        next_id = int(step_last[0].argmax(dim=-1).item())
        tt_logits_list.append(step_last[0])
        tt_token_ids.append(next_id)
        tok_str = tokenizer.decode([next_id])
        top5 = step_last[0].topk(5)
        top5_toks = [tokenizer.decode([t]) for t in top5.indices.tolist()]
        print(
            f"[tt] decode step {step + 1}/{n_decode - 1}: "
            f"time={time.perf_counter() - t_step:.2f}s tok={tok_str!r} "
            f"top5={list(zip(top5_toks, [round(v, 2) for v in top5.values.tolist()]))}",
            flush=True,
        )

    print(
        f"[TT tokens] {[tokenizer.decode([t]) for t in tt_token_ids]}",
        flush=True,
    )
    print(
        f"[TT full]   "
        f"{tokenizer.decode(tokens_single[0].tolist() + tt_token_ids)!r}",
        flush=True,
    )

    # Per-step PCC comparison
    if cpu_logits_list:
        print("", flush=True)
        print("[pcc] per-step logits PCC (TT vs CPU):", flush=True)
        pccs = []
        matches = 0
        for i, (tt_l, cpu_l) in enumerate(zip(tt_logits_list, cpu_logits_list)):
            p = _pcc(tt_l, cpu_l)
            pccs.append(p)
            tt_tok = int(tt_l.argmax().item())
            cpu_tok = int(cpu_l.argmax().item())
            match = tt_tok == cpu_tok
            matches += int(match)
            tag = "OK" if match else "MISMATCH"
            print(
                f"  step {i}: pcc={p:.4f}  tt_tok={tt_tok:>10} "
                f"cpu_tok={cpu_tok:>10}  [{tag}]",
                flush=True,
            )
        print(
            f"[pcc] min={min(pccs):.4f}, max={max(pccs):.4f}, "
            f"avg={sum(pccs) / len(pccs):.4f}",
            flush=True,
        )
        print(f"[tokens] TT==CPU match: {matches}/{len(tt_logits_list)}", flush=True)

    print(f"[timing] total: {time.perf_counter() - t_start:.1f}s", flush=True)
