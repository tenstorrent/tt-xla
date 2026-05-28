# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek v3.1 CPU test, HF-style model.

Drives the canonical HuggingFace-style ``DeepseekV3ForCausalLM`` from
``modeling_deepseek.py``. Loads BF16 weights directly from
``DevQuasar-2/deepseek-ai.DeepSeek-V3.1-BF16`` — no local cache, no FP8
dequant, no ``enable_sparse_mlp`` wrapping (the test runs on CPU).

- ``test_deepseek_v3_1_decode_static_cache``: autoregressive greedy decode
  through ``MLACache`` (``cache_position`` + ``attention_mask``), per-step
  CPU forward.
"""
import json
import os
import re
import sys
import time

import pytest
import torch
from huggingface_hub import hf_hub_download
from infra import MLACache
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(__file__))
from configuration_deepseek import DeepseekV3Config
from modeling_deepseek import DeepseekV3ForCausalLM

DEEPSEEK_V3_1_BF16_REPO = "DevQuasar-2/deepseek-ai.DeepSeek-V3.1-BF16"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_deepseek_config(repo_id=DEEPSEEK_V3_1_BF16_REPO):
    """Download config.json and build a DeepseekV3Config."""
    config_path = hf_hub_download(repo_id, "config.json")
    with open(config_path) as f:
        hf_cfg = json.load(f)
    return DeepseekV3Config(**hf_cfg)


def _load_bf16_weights_direct(model, repo_id, n_layers):
    """Load BF16 weights straight from the HF repo into ``model``.

    Reads the safetensors index, keeps only keys for the first ``n_layers``
    (plus all non-layer keys like embeddings, final norm, lm_head), opens
    each needed shard once, and assigns tensors into the model via
    ``load_state_dict(strict=False, assign=True)``. No on-disk cache is
    written; no dequant runs (the repo is already BF16).
    """
    t0 = time.perf_counter()
    index_path = hf_hub_download(repo_id, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    shard_to_keys = {}
    for ckpt_key, shard_file in weight_map.items():
        layer_m = re.match(r"model\.layers\.(\d+)\.", ckpt_key)
        if layer_m and int(layer_m.group(1)) >= n_layers:
            continue
        shard_to_keys.setdefault(shard_file, []).append(ckpt_key)

    state_dict = {}
    for shard_name, keys in shard_to_keys.items():
        shard_path = hf_hub_download(repo_id, shard_name)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in keys:
                t = f.get_tensor(key)
                if t.dtype != torch.bfloat16:
                    t = t.to(torch.bfloat16)
                state_dict[key] = t

    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    print(
        f"[weights] loaded {len(state_dict)} tensors from {repo_id} in "
        f"{time.perf_counter() - t0:.1f}s. "
        f"Missing: {len(missing)}, Unexpected: {len(unexpected)}",
        flush=True,
    )


def _fix_meta_buffers_hf(model, max_seq_len):
    """Re-materialize non-persistent rotary buffers on CPU after meta init.

    ``DeepseekV3RotaryEmbedding`` registers ``inv_freq``, ``cos_cached``,
    ``sin_cached`` with ``persistent=False`` (modeling_deepseek.py:117,136,137).
    Meta-building the model leaves them meta, and ``load_state_dict`` does not
    restore non-persistent buffers. Force the recompute here on CPU with a
    seq_len at least as large as anything the test will pass at forward time,
    and pin ``max_seq_len_cached`` so the lazy branch never fires.
    """
    for layer in model.model.layers:
        ra = layer.self_attn.rotary_emb
        ra._set_cos_sin_cache(
            seq_len=max_seq_len,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )
        ra.max_seq_len_cached = max_seq_len


def _init_mla_cache(config, batch_size, max_cache_len):
    """Construct an MLACache and eagerly allocate its backing tensors."""
    cache = MLACache(config=config, max_cache_len=max_cache_len)
    for layer in cache.layers:
        layer.lazy_initialization(
            torch.zeros(batch_size, 1, 1, config.kv_lora_rank, dtype=torch.bfloat16),
            torch.zeros(
                batch_size, 1, 1, config.qk_rope_head_dim, dtype=torch.bfloat16
            ),
        )
    return cache


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

    A padding query whose causal window contains only padding keys would
    otherwise have an all-``-inf`` row, producing NaN from softmax. We force
    the diagonal ``(q_pos, q_pos)`` to 0 so every row has at least one finite
    entry; outputs for pad queries are discarded downstream anyway.
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
        )
        mask[:, :, :, :prompt_len] = (
            mask[:, :, :, :prompt_len] + pad_k[:, None, None, :]
        )
    diag_idx = torch.arange(prompt_len)
    mask[:, :, diag_idx, diag_idx] = 0.0
    return mask


def _build_decode_attention_mask(
    batch_size,
    current_pos_inclusive,
    max_cache_len,
    token_attention_mask=None,
    prompt_len=None,
    dtype=torch.bfloat16,
):
    """(B, 1, 1, max_cache_len) additive mask for a single decode step."""
    mask = torch.full((batch_size, 1, 1, max_cache_len), float("-inf"), dtype=dtype)
    mask[:, :, :, :current_pos_inclusive] = 0.0
    if token_attention_mask is not None and prompt_len is not None:
        tam = token_attention_mask.to(dtype=dtype)
        pad_k = torch.where(
            tam > 0,
            torch.zeros((), dtype=dtype),
            torch.full((), float("-inf"), dtype=dtype),
        )
        mask[:, :, :, :prompt_len] = (
            mask[:, :, :, :prompt_len] + pad_k[:, None, None, :]
        )
    return mask


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_layers", [4, 61])
def test_deepseek_v3_1_decode_static_cache(n_layers):
    """Autoregressive greedy CPU decode using the MLACache path."""
    t_start = time.perf_counter()

    from transformers import PreTrainedTokenizerFast

    batch_size = 32
    prompt_seq_len = 32
    n_decode = 10

    repo_id = DEEPSEEK_V3_1_BF16_REPO
    config = load_deepseek_config(repo_id)
    config.num_hidden_layers = n_layers
    # YaRN activates when seq_len exceeds original_max_position_embeddings.
    original_seq_len = config.rope_scaling["original_max_position_embeddings"]
    max_seq_len = original_seq_len + 1  # = 4097, activates YaRN
    print(
        f"[config] num_hidden_layers={config.num_hidden_layers}, "
        f"max_cache_len={max_seq_len}",
        flush=True,
    )

    t0 = time.perf_counter()
    print(f"[timing] config + init: {t0 - t_start:.1f}s", flush=True)

    with torch.device("meta"):
        model = DeepseekV3ForCausalLM(config)
    _load_bf16_weights_direct(model, repo_id, n_layers)
    _fix_meta_buffers_hf(model, max_seq_len)
    model.eval()

    t1 = time.perf_counter()
    print(f"[timing] model build + weight load: {t1 - t0:.1f}s", flush=True)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
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
    token_attn_mask_cpu = encoded["attention_mask"][:, :prompt_seq_len].repeat(
        batch_size, 1
    )
    print(
        f"[prompt] {prompt_len} tokens x{batch_size}: "
        f"{tokenizer.decode(tokens_single[0])!r}",
        flush=True,
    )

    # ===== CPU eager prefill + decode =====
    meta_params = [n for n, p in model.named_parameters() if p.is_meta]
    meta_buffers = [n for n, b in model.named_buffers() if b.is_meta]
    print(
        f"[diag] meta params={len(meta_params)}, meta buffers={len(meta_buffers)}",
        flush=True,
    )
    if meta_params:
        print(f"[diag] first meta params: {meta_params[:5]}", flush=True)
    if meta_buffers:
        print(f"[diag] first meta buffers: {meta_buffers[:5]}", flush=True)
    ra0 = model.model.layers[0].self_attn.rotary_emb
    print(
        f"[diag] rotary[0]: inv_freq.device={ra0.inv_freq.device} "
        f"dtype={ra0.inv_freq.dtype} max_seq_len_cached={ra0.max_seq_len_cached} "
        f"cos_cached.device={ra0.cos_cached.device}",
        flush=True,
    )
    emb_w = model.model.embed_tokens.weight
    print(
        f"[diag] embed_tokens: device={emb_w.device} dtype={emb_w.dtype} "
        f"is_meta={emb_w.is_meta}",
        flush=True,
    )

    print("[cpu] running prefill + decode eagerly...", flush=True)
    cpu_cache = _init_mla_cache(config, batch_size, max_seq_len)
    cpu_token_ids = []
    t_cpu_start = time.perf_counter()
    with torch.no_grad():
        attn_mask = _build_prefill_attention_mask(
            batch_size,
            prompt_len,
            max_seq_len,
            token_attention_mask=token_attn_mask_cpu,
        )
        out = model(
            input_ids=tokens_cpu,
            attention_mask=attn_mask,
            past_key_values=cpu_cache,
            cache_position=torch.arange(prompt_len, dtype=torch.long),
            use_cache=True,
            return_dict=True,
        )
        _l = out.logits.float()
        print(
            f"[diag] prefill logits: shape={tuple(out.logits.shape)} "
            f"dtype={out.logits.dtype} "
            f"mean={_l.mean().item():.4f} std={_l.std().item():.4f} "
            f"absmax={_l.abs().max().item():.4f} "
            f"nan%={torch.isnan(out.logits).float().mean().item():.4f} "
            f"inf%={torch.isinf(out.logits).float().mean().item():.4f}",
            flush=True,
        )
        next_id = int(out.logits[0, -1].argmax(dim=-1).item())
        cpu_token_ids.append(next_id)
        for step in range(n_decode - 1):
            pos = prompt_len + step
            next_tokens = torch.full((batch_size, 1), next_id, dtype=tokens_cpu.dtype)
            attn_mask = _build_decode_attention_mask(
                batch_size,
                pos + 1,
                max_seq_len,
                token_attention_mask=token_attn_mask_cpu,
                prompt_len=prompt_len,
            )
            out = model(
                input_ids=next_tokens,
                attention_mask=attn_mask,
                past_key_values=cpu_cache,
                cache_position=torch.tensor([pos], dtype=torch.long),
                use_cache=True,
                return_dict=True,
            )
            next_id = int(out.logits[0, -1].argmax(dim=-1).item())
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
