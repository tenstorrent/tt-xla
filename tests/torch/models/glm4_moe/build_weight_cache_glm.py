#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""GLM-4.7 weight cache: stacked-experts layout, BF16 throughout.

GLM-4.7 is already BF16 on HF, so this is a single-stage cache. The chunked
layout (one safetensors file per layer plus `shared.safetensors`) and the
build orchestration live in `tests/infra/weight_cache/`; this module just
declares the per-model spec:

- `_iter_glm_groups` decides which HF keys go into which output chunk
  (shared / dense / moe layer).
- `_transform_glm_group` does the per-chunk work: cast to BF16, and for MoE
  layers stack per-expert tensors into the `[E, in, out]` layout consumed by
  `enable_sparse_mlp` plus rename the router's gate keys.

Run as a script to build the cache from the command line:

    python build_weight_cache_glm.py --n-layers 4    # smoke / bringup
    python build_weight_cache_glm.py --n-layers 92   # full GLM-4.7
"""
import argparse
import json
import re

import torch
from huggingface_hub import hf_hub_download
from infra.weight_cache import (
    GroupDef,
    WeightCacheSpec,
    build_cache,
    cache_dir_for,
    safe_open_hf,
)

GLM_REPO = "zai-org/GLM-4.7"

# Per-expert weight key shape (one MoE layer). Layer index is injected at
# format time so each call site gets a tightly-anchored regex.
_EXPERT_RE_TPL = (
    r"^model\.layers\.{layer_idx}\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.weight$"
)


def _iter_glm_groups(weight_map, *, n_layers, n_dense_layers, n_experts):
    """Yield one `shared` group + one group per layer (dense or moe)."""
    shared_keys = [
        k
        for k in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight")
        if k in weight_map
    ]
    yield GroupDef(name="shared", ckpt_keys=shared_keys, metadata={"type": "shared"})

    for i in range(n_layers):
        prefix = f"model.layers.{i}."
        layer_keys = [k for k in weight_map if k.startswith(prefix)]
        yield GroupDef(
            name=f"layer_{i:04d}",
            ckpt_keys=layer_keys,
            metadata={
                "type": "dense" if i < n_dense_layers else "moe",
                "layer_idx": i,
                "n_experts": n_experts,
            },
        )


def _transform_glm_group(raw, group):
    """Dispatch on group type. Shared/dense layers pass through; MoE stacks."""
    group_type = group.metadata["type"]
    if group_type in ("shared", "dense"):
        return {k: raw[k].to(torch.bfloat16) for k in group.ckpt_keys}
    if group_type == "moe":
        return _transform_moe_layer(raw, group)
    raise ValueError(f"unknown GLM group type: {group_type!r}")


def _transform_moe_layer(raw, group):
    """Stack per-expert tensors into `[E, in, out]`; rename router keys."""
    layer_idx = group.metadata["layer_idx"]
    n_experts = group.metadata["n_experts"]
    prefix = f"model.layers.{layer_idx}"
    expert_re = re.compile(_EXPERT_RE_TPL.format(layer_idx=layer_idx))

    expert_keys = [k for k in group.ckpt_keys if expert_re.match(k)]
    other_keys = [k for k in group.ckpt_keys if not expert_re.match(k)]

    # HF stores each expert as `nn.Linear` convention `[out, in]`. The TT
    # StackedExperts layout is `[E, in, out]`, so transpose at stack time.
    gate_list: list[torch.Tensor | None] = [None] * n_experts
    up_list: list[torch.Tensor | None] = [None] * n_experts
    down_list: list[torch.Tensor | None] = [None] * n_experts
    for k in expert_keys:
        m = expert_re.match(k)
        idx = int(m.group(1))
        name = m.group(2)
        t = raw[k].to(torch.bfloat16).T.contiguous()
        if name == "gate_proj":
            gate_list[idx] = t
        elif name == "up_proj":
            up_list[idx] = t
        elif name == "down_proj":
            down_list[idx] = t

    missing = [i for i, x in enumerate(gate_list) if x is None]
    if missing:
        raise ValueError(
            f"Layer {layer_idx}: missing gate_proj for experts {missing[:5]}..."
        )

    out = {
        f"{prefix}.mlp.mlp.experts.gate_proj": torch.stack(gate_list, dim=0),
        f"{prefix}.mlp.mlp.experts.up_proj": torch.stack(up_list, dim=0),
        f"{prefix}.mlp.mlp.experts.down_proj": torch.stack(down_list, dim=0),
    }
    del gate_list, up_list, down_list

    router_weight_key = f"{prefix}.mlp.gate.weight"
    router_bias_key = f"{prefix}.mlp.gate.e_score_correction_bias"
    for k in other_keys:
        t = raw[k].to(torch.bfloat16)
        if k == router_weight_key:
            out[f"{prefix}.mlp.mlp.router.gate.weight"] = t
        elif k == router_bias_key:
            out[f"{prefix}.mlp.mlp.router.gate.e_score_correction_bias"] = t
        else:
            out[k] = t

    return out


def glm_weight_cache_spec(
    repo_id: str, n_layers: int, n_dense_layers: int, n_experts: int
) -> WeightCacheSpec:
    """Build the WeightCacheSpec for a GLM-4.7-compatible config."""
    return WeightCacheSpec(
        repo_id=repo_id,
        cache_dir=cache_dir_for(repo_id, n_layers, variant="stacked"),
        iter_groups=lambda weight_map: _iter_glm_groups(
            weight_map,
            n_layers=n_layers,
            n_dense_layers=n_dense_layers,
            n_experts=n_experts,
        ),
        transform_group=_transform_glm_group,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build weight cache for GLM-4.7 (stacked-experts variant)"
    )
    parser.add_argument("--repo", default=GLM_REPO, help="HuggingFace repo ID")
    parser.add_argument("--n-layers", type=int, default=92, help="Number of layers")
    args = parser.parse_args()

    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*", args.repo
    ):
        parser.error(f"Invalid repo ID {args.repo!r}: expected 'org/model' format")

    config_path = hf_hub_download(args.repo, "config.json")
    with safe_open_hf(config_path) as f:
        cfg = json.load(f)
    n_dense_layers = cfg["first_k_dense_replace"]
    n_experts = cfg["n_routed_experts"]
    print(
        f"Repo: {args.repo}\n  n_layers={args.n_layers}, "
        f"n_dense_layers={n_dense_layers}, n_experts={n_experts}",
        flush=True,
    )

    build_cache(
        glm_weight_cache_spec(args.repo, args.n_layers, n_dense_layers, n_experts)
    )
