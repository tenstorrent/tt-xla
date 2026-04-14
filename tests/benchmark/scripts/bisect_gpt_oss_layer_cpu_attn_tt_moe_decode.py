#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Bisect GPT-OSS **decode** step: one decoder layer uses CPU attention + TT MoE.

For **prefill**-phase hybrid (full sequence through CPU attn + TT MoE at layer L), use
``bisect_gpt_oss_layer_cpu_attn_tt_moe_prefill.py``.

Mirrors ``benchmark_llm_torch_xla`` decode-only setup:

1. **CPU prefill** — fills ``StaticCache`` (``build_cpu_decode_start_after_prefill``).
2. **CPU reference decode** — one forward with first token + ``cache_position``; optional
   snapshot/restore of KV so cache matches prefill end (same as PCC path in
   ``llm_benchmark``).
3. **Hybrid decode** — run part of layer ``L`` on TT, rest on CPU (see ``--tt-scope``).

**Finer than a full layer**

* ``--tt-scope mlp`` — only the MoE block (``post_attention_layernorm`` still on CPU, same
  as swapping ``layer.mlp``).
* ``--tt-scope post_norm_mlp`` — **RMSNorm after attention + MoE + residual add** on TT in
  one compiled graph (attention and first half of the layer stay on CPU). Isolates whether
  bugs sit in post-norm vs MoE vs the add.
* **Finer still** — per-op replay from TTNN dumps: ``test_topk_to_scatter_pipeline.py`` /
  ``test_topk_gpt_oss.py``; attention projections on TT would need a separate compiled
  ``self_attn`` subgraph and Galaxy shard specs for Q/K/V/O (not wired here).

**Important:** Final ``lm_head`` logits are only comparable to a full-CPU decode when the
model **ends right after** layer ``L`` (otherwise later CPU layers would see activations
produced by TT MoE at ``L``, which breaks the CPU reference). By default this script sets
``num_layers = layer + 1``. Override ``--num-layers`` only if you accept that logits will
then diverge unless the TT MoE is bit-identical.

**Suggested order** with ``gpt_oss_galaxy_layer_diagnose.py`` (device idle between steps):

1. ``baseline`` — CPU prefill + TT decode vs CPU golden (end-to-end decode PCC).
2. ``prefill_check`` — TT vs CPU **prefill-only** last-position logits (no decode).
3. ``kv_tt_prefill`` — TT prefill writes KV, then TT decode (split KV transfer vs on-device KV).
4. ``bisect_gpt_oss_layer_cpu_attn_tt_moe_prefill.py`` — hybrid during **prefill** (long seq).
5. This script — hybrid during **decode** only (``--tt-scope``).

Run from ``tests/benchmark``::

    cd tests/benchmark
    python scripts/bisect_gpt_oss_layer_cpu_attn_tt_moe_decode.py --layer 0

Requires 32 TT devices (wormhole_galaxy), same env as ``test_gpt_oss_20b_tp_galaxy_batch_size_64``.
``--batch-size`` must be 64; shardings match the Galaxy 20B benchmark spec in test_llms.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr

_BENCH_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BENCH_ROOT.parent.parent
sys.path.insert(0, str(_BENCH_ROOT))
sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.llm_benchmark import (  # noqa: E402
    DEFAULT_INPUT_PROMPT,
    MODULE_EXPORT_PATH,
    build_cpu_decode_start_after_prefill,
    check_transformers_version,
    get_mesh,
    setup_model_and_tokenizer,
)
from llm_utils.gpt_oss_galaxy_subnet_sharding import (  # noqa: E402
    mark_gpt_oss_galaxy_copied_subnet_weights,
)
from llm_utils.gpt_oss_kv_debug import debug_kv_report  # noqa: E402
from test_llms import (  # noqa: E402
    _gpt_oss_galaxy_mesh_config_fn,
    _gpt_oss_galaxy_shard_spec_fn,
)
from utils import build_xla_export_name, compute_pcc_flat_pair, create_model_loader  # noqa: E402

xr.set_device_type("TT")

_GALAXY_GPT_OSS_20B_BENCHMARK_BATCH_SIZE = 64


def _ensure_galaxy_mesh(num_devices: int) -> None:
    try:
        _gpt_oss_galaxy_mesh_config_fn(None, num_devices)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _mlp_out_hidden(mlp: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    out = mlp(hidden_states)
    return out[0] if isinstance(out, tuple) else out


class _MoeSubnet(nn.Module):
    def __init__(self, mlp: nn.Module):
        super().__init__()
        self.mlp = mlp

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return _mlp_out_hidden(self.mlp, hidden_states)


class _PostNormMoeSubnet(nn.Module):
    """``residual + mlp(post_attention_layernorm(x))`` where ``x`` is the post-attn residual stream."""

    def __init__(self, post_norm: nn.Module, mlp: nn.Module):
        super().__init__()
        self.post_norm = post_norm
        self.mlp = mlp

    def forward(self, hidden_after_attn_residual: torch.Tensor) -> torch.Tensor:
        residual = hidden_after_attn_residual
        normed = self.post_norm(hidden_after_attn_residual)
        mlp_out = _mlp_out_hidden(self.mlp, normed)
        return residual + mlp_out


class _TtMlpAsCpuModule(nn.Module):
    """Swapped in as ``layer.mlp``: run compiled MoE on TT; tuple matches GPT-OSS ``mlp`` API."""

    def __init__(self, compiled: torch.nn.Module, mesh, device: torch.device):
        super().__init__()
        self._compiled = compiled
        self._mesh = mesh
        self._device = device

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
        x = hidden_states.to(device=self._device, dtype=torch.bfloat16)
        xs.mark_sharding(x, self._mesh, ("batch", None, None))
        y = self._compiled(x)
        xm.mark_step()
        y_cpu = y.detach().cpu().to(torch.bfloat16)
        return (y_cpu, None)


class _TtPostNormMlpAsCpuModule(nn.Module):
    """Runs compiled ``post_norm + moe + residual`` on TT; returns CPU bf16 layer tail output."""

    def __init__(self, compiled: nn.Module, mesh, device: torch.device):
        super().__init__()
        self._compiled = compiled
        self._mesh = mesh
        self._device = device

    def forward(self, hidden_after_attn_residual: torch.Tensor) -> torch.Tensor:
        x = hidden_after_attn_residual.to(device=self._device, dtype=torch.bfloat16)
        xs.mark_sharding(x, self._mesh, ("batch", None, None))
        y = self._compiled(x)
        xm.mark_step()
        return y.detach().cpu().to(torch.bfloat16)


def _hybrid_decoder_layer_forward_post_norm_mlp(
    self,
    hidden_states: torch.Tensor,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    use_cache: bool = False,
    cache_position=None,
    position_embeddings=None,
    *,
    tt_tail: nn.Module,
    **kwargs,
):
    """Match ``GptOssDecoderLayer.forward``; post-norm + MoE + residual on TT via ``tt_tail``."""

    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, _ = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = residual + hidden_states
    return tt_tail(hidden_states)


def _min_pcc_logits(golden: torch.Tensor, actual: torch.Tensor) -> float:
    if golden.shape != actual.shape:
        raise ValueError(f"logits shape {tuple(actual.shape)} vs golden {tuple(golden.shape)}")
    pccs = []
    for b in range(golden.shape[0]):
        pccs.append(compute_pcc_flat_pair(golden[b, -1], actual[b, -1]))
    return min(pccs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, default=0, help="Decoder layer index (MoE on TT).")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=_GALAXY_GPT_OSS_20B_BENCHMARK_BATCH_SIZE,
        help=(
            f"Must be {_GALAXY_GPT_OSS_20B_BENCHMARK_BATCH_SIZE} "
            "(``test_gpt_oss_20b_tp_galaxy_batch_size_64``)."
        ),
    )
    parser.add_argument("--isl", type=int, default=128, help="Max cache / prompt length (prefill).")
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help=(
            "Truncate model depth. Default: layer+1 so logits compare cleanly (no layers after L). "
            "Larger values keep deeper CPU layers but logits vs full-CPU decode will not match unless "
            "TT MoE matches CPU exactly."
        ),
    )
    parser.add_argument("--required-pcc", type=float, default=0.95)
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument(
        "--experimental-weight-dtype",
        type=str,
        default="bfp_bf8",
        help="Compile option (match test_llm).",
    )
    parser.add_argument(
        "--tt-scope",
        choices=("mlp", "post_norm_mlp"),
        default="mlp",
        help=(
            "mlp: only MoE on TT (post_attention_layernorm on CPU). "
            "post_norm_mlp: post_attention_layernorm + MoE + residual add on TT in one graph."
        ),
    )
    args = parser.parse_args()

    check_transformers_version()
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()

    num_dev = xr.global_runtime_device_count()
    _ensure_galaxy_mesh(num_dev)
    if not (args.batch_size == _GALAXY_GPT_OSS_20B_BENCHMARK_BATCH_SIZE):
        raise SystemExit(
            f"--batch-size {args.batch_size} must be 64 for test_gpt_oss_20b_tp_galaxy_batch_size_64; see --batch-size help."
        )

    from third_party.tt_forge_models.gpt_oss.pytorch.loader import ModelLoader, ModelVariant

    effective_num_layers = (
        args.num_layers if args.num_layers is not None else args.layer + 1
    )
    if args.layer >= effective_num_layers:
        raise SystemExit(
            f"--layer {args.layer} must be < effective num_layers ({effective_num_layers})."
        )
    if args.num_layers is not None and args.num_layers > args.layer + 1:
        print(
            "[bisect] warning: num_layers > layer+1 — logits vs full-CPU decode reference mix "
            "TT MoE output at L into later CPU layers; PCC may be low even if MoE is OK. "
            "Omit --num-layers for default (layer+1)."
        )

    model_loader = create_model_loader(
        ModelLoader, num_layers=effective_num_layers, variant=ModelVariant.GPT_OSS_20B
    )
    if model_loader is None:
        raise SystemExit("ModelLoader does not support num_layers override.")

    mesh = get_mesh(model_loader, _gpt_oss_galaxy_mesh_config_fn)
    device = torch_xla.device()

    model, tokenizer = setup_model_and_tokenizer(model_loader, ModelVariant.GPT_OSS_20B)
    n_layers = len(model.model.layers)
    if args.layer < 0 or args.layer >= n_layers:
        raise SystemExit(f"--layer must be in [0, {n_layers - 1}]")

    layer = model.model.layers[args.layer]
    orig_mlp = layer.mlp
    orig_forward = type(layer).forward
    mlp_tt = copy.deepcopy(orig_mlp).to(device=device, dtype=torch.bfloat16)
    mlp_tt.eval()

    if args.tt_scope == "mlp":
        subnet = _MoeSubnet(mlp_tt).eval()
        mark_gpt_oss_galaxy_copied_subnet_weights(
            model,
            model_loader,
            args.layer,
            mesh,
            mlp_tt,
            None,
            _gpt_oss_galaxy_shard_spec_fn,
        )
    else:
        post_tt = copy.deepcopy(layer.post_attention_layernorm).to(
            device=device, dtype=torch.bfloat16
        )
        post_tt.eval()
        subnet = _PostNormMoeSubnet(post_tt, mlp_tt).eval()
        mark_gpt_oss_galaxy_copied_subnet_weights(
            model,
            model_loader,
            args.layer,
            mesh,
            mlp_tt,
            post_tt,
            _gpt_oss_galaxy_shard_spec_fn,
        )

    export_name = build_xla_export_name(
        model_name=f"gpt_oss_hybrid_decode_L{args.layer}_{args.tt_scope}",
        num_layers=effective_num_layers,
        batch_size=args.batch_size,
        input_sequence_length=args.isl,
    )
    torch_xla.set_custom_compile_options(
        {
            "optimization_level": args.optimization_level,
            "enable_trace": False,
            "export_path": MODULE_EXPORT_PATH,
            "export_model_name": export_name,
            "ttnn_perf_metrics_enabled": False,
            "experimental_weight_dtype": args.experimental_weight_dtype,
            "experimental_enable_permute_matmul_fusion": False,
        }
    )
    torch._dynamo.reset()
    compiled = torch.compile(subnet, backend="tt")
    if args.tt_scope == "mlp":
        tt_hybrid_mod = _TtMlpAsCpuModule(compiled, mesh, device)
    else:
        tt_hybrid_mod = _TtPostNormMlpAsCpuModule(compiled, mesh, device)

    def read_logits(o):
        return o.logits

    input_args, golden_logits, prefill_len = build_cpu_decode_start_after_prefill(
        model=model,
        tokenizer=tokenizer,
        model_config=model.config,
        batch_size=args.batch_size,
        max_cache_len=args.isl,
        custom_input_prompt=DEFAULT_INPUT_PROMPT,
        input_prompt_tokens=None,
        read_logits_fn=read_logits,
        accuracy_testing=False,
        token_accuracy=None,
        capture_cpu_decode_logits_for_pcc=True,
    )
    print(
        f"[bisect] prefill_len={prefill_len} decode input_ids shape={tuple(input_args['input_ids'].shape)} "
        f"golden logits shape={tuple(golden_logits.shape)}"
    )
    debug_kv_report(
        input_args["past_key_values"],
        "bisect_after_cpu_prefill_kv_on_cpu",
        flush_xla=False,
    )

    try:
        if args.tt_scope == "mlp":
            layer.mlp = tt_hybrid_mod
        else:
            _tt = tt_hybrid_mod

            def _fwd(
                self,
                hidden_states,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                use_cache=False,
                cache_position=None,
                position_embeddings=None,
                **kwargs,
            ):
                return _hybrid_decoder_layer_forward_post_norm_mlp(
                    self,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    tt_tail=_tt,
                    **kwargs,
                )

            layer.forward = types.MethodType(_fwd, layer)

        with torch.no_grad():
            hybrid_out = model(
                input_ids=input_args["input_ids"],
                past_key_values=input_args["past_key_values"],
                cache_position=input_args["cache_position"],
                use_cache=True,
            )
    finally:
        if args.tt_scope == "mlp":
            layer.mlp = orig_mlp
        else:
            layer.forward = types.MethodType(orig_forward, layer)

    hybrid_logits = read_logits(hybrid_out)
    pcc_min = _min_pcc_logits(golden_logits, hybrid_logits)
    print(
        f"[bisect] layer={args.layer} tt_scope={args.tt_scope} "
        f"logits PCC (last token, min over batch)={pcc_min:.6f} "
        f"required={args.required_pcc:.6f}"
    )
    if pcc_min < args.required_pcc:
        raise SystemExit(
            f"PCC below threshold: hybrid decode (tt_scope={args.tt_scope!r} at L{args.layer}) "
            f"differs from full CPU decode."
        )
    print("[bisect] OK — hybrid decode logits match CPU reference at required PCC.")


if __name__ == "__main__":
    main()
