#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V3.1 / V3.2-exp weight cache builder.

Two-stage cache used by the DeepSeek test files:

- `_bf16`: BF16 weights after FP8 block-wise dequant and the HF -> modified_model
  key rename. Useful as a CPU reference (per-expert weights still live here).
- `_stacked`: post-sparse cache where each MoE chunk has its per-expert
  `w1/w2/w3` tensors stacked into the StackedExperts layout (with zero biases)
  consumed by `A2aSparseMLP`.

The chunked layout (one .safetensors per HF-layer plus `shared.safetensors`) and
the build orchestration live in `tests/infra/weight_cache/`; this module declares
the per-model spec:

- `_iter_deepseek_bf16_groups` decides which HF keys go into which output chunk,
  attaches the matching `*.weight_scale_inv` aux keys, and carries the
  ckpt→model rename map in `GroupDef.metadata`.
- `_transform_deepseek_bf16_group` dequantizes FP8, casts to BF16 (or FP32 for
  `head.weight`), and writes each tensor under its model key.
- `_transform_deepseek_post_sparse_chunk` detects MoE chunks by key shape and
  applies `_stack_experts_for_chunk` (single-pass over the bf16 chunk).

Run as a script to build either stage from the command line:

    python build_weight_cache.py                   # default: 61 layers, _bf16 only
    python build_weight_cache.py --n-layers 4      # smoke
    python build_weight_cache.py --post-sparse     # also build _stacked
"""
import argparse
import json
import re

import torch
from huggingface_hub import hf_hub_download
from infra.weight_cache import (
    GroupDef,
    WeightCacheSpec,
    cache_dir_for,
    ensure_cache,
    maybe_dequant,
    safe_open_hf,
)

DEEPSEEK_V3_1_REPO = "deepseek-ai/DeepSeek-V3.1"


def _rename_hf_key(ckpt_key, n_dense_layers=1):
    """Rename a HuggingFace checkpoint key to match `modified_model.py` naming.

    Returns `None` if the key should be dropped: FP8 scale tensors are tracked
    separately via aux keys, and `mlp.gate_proj/up_proj/down_proj` on layers
    >= `n_dense_layers` are MoE-only and don't exist in the modified model.
    """
    key = ckpt_key
    if key.startswith("model."):
        key = key[len("model.") :]
    if "weight_scale_inv" in key:
        return None
    key = key.replace("lm_head.", "head.")
    key = key.replace("embed_tokens.", "embed.")
    key = re.sub(r"(layers\.\d+\.)input_layernorm\.", r"\1attn_norm.", key)
    key = re.sub(r"(layers\.\d+\.)post_attention_layernorm\.", r"\1ffn_norm.", key)
    key = key.replace("self_attn.indexer.", "attn.indexer.")
    key = key.replace("self_attn.q_a_proj.", "attn.wq_a.")
    key = key.replace("self_attn.q_b_proj.", "attn.wq_b.")
    key = key.replace("self_attn.q_a_layernorm.", "attn.q_norm.")
    key = key.replace("self_attn.kv_a_proj_with_mqa.", "attn.wkv_a.")
    key = key.replace("self_attn.kv_b_proj.", "attn.wkv_b.")
    key = key.replace("self_attn.kv_a_layernorm.", "attn.kv_norm.")
    key = key.replace("self_attn.o_proj.", "attn.wo.")
    key = re.sub(r"mlp\.experts\.(\d+)\.gate_proj\.", r"ffn.experts.\1.w1.", key)
    key = re.sub(r"mlp\.experts\.(\d+)\.down_proj\.", r"ffn.experts.\1.w2.", key)
    key = re.sub(r"mlp\.experts\.(\d+)\.up_proj\.", r"ffn.experts.\1.w3.", key)
    key = key.replace("mlp.shared_experts.gate_proj.", "ffn.shared_experts.w1.")
    key = key.replace("mlp.shared_experts.down_proj.", "ffn.shared_experts.w2.")
    key = key.replace("mlp.shared_experts.up_proj.", "ffn.shared_experts.w3.")
    key = key.replace("mlp.gate.e_score_correction_bias", "mlp.gate.bias")
    key = key.replace("mlp.gate.", "ffn.gate.")
    layer_m = re.match(r"layers\.(\d+)\.", key)
    if layer_m:
        layer_id = int(layer_m.group(1))
        if layer_id < n_dense_layers:
            key = key.replace("mlp.gate_proj.", "ffn.w1.")
            key = key.replace("mlp.down_proj.", "ffn.w2.")
            key = key.replace("mlp.up_proj.", "ffn.w3.")
        elif (
            "mlp.gate_proj." in key or "mlp.down_proj." in key or "mlp.up_proj." in key
        ):
            return None
    return key


def _build_deepseek_groups(weight_map, n_layers, n_dense_layers):
    """Partition HF keys into output groups + scale-key sidetable.

    Returns `(groups, scale_keys)` where:
    - `groups`: ordered dict `{group_name: {ckpt_key: model_key}}`
    - `scale_keys`: `{ckpt_weight_key: ckpt_scale_key}` for FP8 weights that
      have a paired `*.weight_scale_inv` tensor on disk
    """
    groups: dict[str, dict[str, str]] = {}
    scale_keys: dict[str, str] = {}

    for ckpt_key in weight_map:
        layer_m = re.match(r"model\.layers\.(\d+)\.", ckpt_key)
        if layer_m and int(layer_m.group(1)) >= n_layers:
            continue

        if "weight_scale_inv" in ckpt_key:
            w_key = ckpt_key.replace(".weight_scale_inv", ".weight")
            scale_keys[w_key] = ckpt_key
            continue

        model_key = _rename_hf_key(ckpt_key, n_dense_layers)
        if model_key is None:
            continue

        lm = re.match(r"layers\.(\d+)\.", model_key)
        group_name = f"layer_{int(lm.group(1)):04d}" if lm else "shared"
        groups.setdefault(group_name, {})[ckpt_key] = model_key

    return groups, scale_keys


def _iter_deepseek_bf16_groups(weight_map, *, n_layers, n_dense_layers):
    groups, scale_keys = _build_deepseek_groups(weight_map, n_layers, n_dense_layers)
    for group_name in sorted(groups):
        ckpt_to_model = groups[group_name]
        ckpt_keys = list(ckpt_to_model.keys())
        aux_keys = [scale_keys[k] for k in ckpt_keys if k in scale_keys]
        yield GroupDef(
            name=group_name,
            ckpt_keys=ckpt_keys,
            aux_keys=aux_keys,
            metadata={
                "ckpt_to_model": ckpt_to_model,
                "scale_keys": {k: scale_keys[k] for k in ckpt_keys if k in scale_keys},
            },
        )


def _transform_deepseek_bf16_group(raw, group):
    ckpt_to_model = group.metadata["ckpt_to_model"]
    scale_keys = group.metadata["scale_keys"]

    out: dict[str, torch.Tensor] = {}
    for ckpt_key, model_key in ckpt_to_model.items():
        tensor = raw.get(ckpt_key)
        if tensor is None:
            continue
        scale_key = scale_keys.get(ckpt_key)
        scale = raw.get(scale_key) if scale_key else None
        tensor = maybe_dequant(tensor, scale)
        if model_key == "head.weight":
            tensor = tensor.to(torch.float32)
        elif tensor.dtype != torch.bfloat16:
            tensor = tensor.to(torch.bfloat16)
        out[model_key] = tensor
    return out


def _stack_experts_for_chunk(chunk):
    """Convert per-expert weights in one bf16 chunk to the StackedExperts layout.

    Input keys `layers.N.ffn.experts.{idx}.{w1,w2,w3}.weight` are stacked +
    transposed into `layers.N.ffn.mlp.experts.{gate,up,down}_proj` plus zero
    `*_proj_bias`. Router keys `ffn.gate.*` become `ffn.mlp.router.gate.*`.
    Everything else (attention, norms, shared_experts) passes through.
    """
    expert_weights: dict[int, dict[str, torch.Tensor]] = {}
    layer_prefix: str | None = None

    for key in chunk:
        m = re.match(r"(layers\.\d+\.ffn)\.experts\.(\d+)\.(w[123])\.weight", key)
        if m:
            layer_prefix = m.group(1)
            idx, wname = int(m.group(2)), m.group(3)
            expert_weights.setdefault(idx, {})[wname] = chunk[key]

    if not expert_weights:
        return dict(chunk)

    n_experts = max(expert_weights.keys()) + 1
    result: dict[str, torch.Tensor] = {}

    # w1=gate, w3=up, w2=down — transpose at stack time to get [E, in, out].
    gate_proj = torch.stack([expert_weights[i]["w1"].T for i in range(n_experts)])
    up_proj = torch.stack([expert_weights[i]["w3"].T for i in range(n_experts)])
    down_proj = torch.stack([expert_weights[i]["w2"].T for i in range(n_experts)])

    inter = gate_proj.shape[-1]
    hidden = gate_proj.shape[1]
    dtype = gate_proj.dtype

    mlp_pfx = f"{layer_prefix}.mlp.experts"
    result[f"{mlp_pfx}.gate_proj"] = gate_proj
    result[f"{mlp_pfx}.up_proj"] = up_proj
    result[f"{mlp_pfx}.down_proj"] = down_proj
    result[f"{mlp_pfx}.gate_proj_bias"] = torch.zeros(n_experts, inter, dtype=dtype)
    result[f"{mlp_pfx}.up_proj_bias"] = torch.zeros(n_experts, inter, dtype=dtype)
    result[f"{mlp_pfx}.down_proj_bias"] = torch.zeros(n_experts, hidden, dtype=dtype)

    for key in chunk:
        if ".ffn.experts." in key:
            continue
        if ".ffn.gate." in key:
            result[key.replace(".ffn.gate.", ".ffn.mlp.router.gate.")] = chunk[key]
        else:
            result[key] = chunk[key]

    return result


def _transform_deepseek_post_sparse_chunk(chunk, chunk_name):
    """Apply expert stacking iff the chunk has MoE expert keys."""
    if any(".ffn.experts." in k for k in chunk):
        return _stack_experts_for_chunk(chunk)
    return dict(chunk)


def deepseek_weight_cache_spec(
    repo_id: str,
    n_layers: int,
    n_dense_layers: int = 1,
    *,
    post_sparse: bool = False,
) -> WeightCacheSpec:
    """Build the WeightCacheSpec for DeepSeek V3.1 / V3.2-exp.

    With `post_sparse=False`, returns the BF16 spec (HF-source, dequant +
    rename). With `post_sparse=True`, returns a stacked spec whose
    `next_stage` is the BF16 spec — `ensure_cache` will materialize both.
    """
    bf16 = WeightCacheSpec(
        repo_id=repo_id,
        cache_dir=cache_dir_for(repo_id, n_layers, variant="bf16"),
        iter_groups=lambda weight_map: _iter_deepseek_bf16_groups(
            weight_map, n_layers=n_layers, n_dense_layers=n_dense_layers
        ),
        transform_group=_transform_deepseek_bf16_group,
    )
    if not post_sparse:
        return bf16
    return WeightCacheSpec(
        repo_id=repo_id,
        cache_dir=cache_dir_for(repo_id, n_layers, variant="stacked"),
        transform_chunk=_transform_deepseek_post_sparse_chunk,
        next_stage=bf16,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build weight cache for DeepSeek V3.1 / V3.2-exp"
    )
    parser.add_argument(
        "--repo", default=DEEPSEEK_V3_1_REPO, help="HuggingFace repo ID"
    )
    parser.add_argument("--n-layers", type=int, default=61, help="Number of layers")
    parser.add_argument(
        "--n-dense-layers",
        type=int,
        default=None,
        help="Number of dense (non-MoE) layers (default: read from config)",
    )
    parser.add_argument(
        "--post-sparse",
        action="store_true",
        help="Also build the post-sparse (stacked-experts) stage",
    )
    args = parser.parse_args()

    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*", args.repo
    ):
        parser.error(f"Invalid repo ID {args.repo!r}: expected 'org/model' format")

    n_dense_layers = args.n_dense_layers
    if n_dense_layers is None:
        config_path = hf_hub_download(args.repo, "config.json")
        with safe_open_hf(config_path) as f:
            hf_cfg = json.load(f)
        n_dense_layers = hf_cfg["first_k_dense_replace"]
        print(f"Read n_dense_layers={n_dense_layers} from {args.repo} config")

    spec = deepseek_weight_cache_spec(
        args.repo,
        args.n_layers,
        n_dense_layers,
        post_sparse=args.post_sparse,
    )
    ensure_cache(spec)
