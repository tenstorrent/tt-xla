# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Incremental layer-0 ``input_layernorm`` → ``LlamaAttention`` bisection (issue #4968).

**Graph fidelity (important)**

- **Canonical 0.77 repro:** ``JanusLlamaDecoderLayer0LnAttnProfile`` returns
  ``torch.stack([decode_setup, input_layernorm, self_attn_out])``. That **multi-output
  fused graph** is what drops to ~**0.77** on the ``self_attn`` slice (see
  ``test_image_token_decode_decoder_layer0_input_layernorm_self_attn_loader_pro_1b``).
- **Bisect ``native_self_attn`` (~0.99):** same PyTorch ops, but returns **only**
  ``attn_output`` (single output). TT compiles a **different graph** — does **not**
  reproduce the 0.77 bug. Do not use for tt-metal / TTNN investigation.
- Stages ``q_proj`` … ``o_proj`` also use **different** graphs (decomposed attention).

**KV cache**

``past_key_values.update`` mutates in-place. Each ``forward`` clones the module cache via
``prepare_decode_past_key_values`` so ``run_op_test`` (CPU then TT) and multi-stage
profiles always start from the same prefill snapshot (fixes 51 vs 52 seq mismatch).

Each stage compiles **one TT graph** and compares the stage output tensor vs CPU golden.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from tests.torch.models.janus_pro_pcc_drop.decoder_op_test_utils import (
    TensorMatchMetrics,
    print_tensor_match_metrics,
    run_decoder_op_test_collect_metrics,
)
from tests.torch.models.janus_pro_pcc_drop.decoder_sanity import _llama_decode_step_tensors
from tests.torch.models.janus_pro_pcc_drop.decoder_submodule_sanity import (
    DEFAULT_DECODE_LAYER_IDX,
    JanusLlamaDecoderLayer0LnAttnProfile,
    prepare_decode_past_key_values,
)
from infra import Framework, run_op_test
from infra.evaluators import ComparisonConfig, TorchComparisonEvaluator


@dataclass(frozen=True)
class LnAttnBisectStageSpec:
    stage_id: str
    label: str
    ops_added: str
    model_faithful: bool


LAYER0_LN_ATTN_BISECT_STAGES: tuple[LnAttnBisectStageSpec, ...] = (
    LnAttnBisectStageSpec(
        "layernorm", "01_layernorm", "input_layernorm", model_faithful=True
    ),
    LnAttnBisectStageSpec("q_proj", "02_ln_q_proj", "q_proj", model_faithful=False),
    LnAttnBisectStageSpec("k_proj", "03_ln_qk_proj", "k_proj", model_faithful=False),
    LnAttnBisectStageSpec("v_proj", "04_ln_qkv_proj", "v_proj", model_faithful=False),
    LnAttnBisectStageSpec(
        "rope", "05_ln_qkv_rope", "apply_rotary_pos_emb", model_faithful=False
    ),
    LnAttnBisectStageSpec(
        "kv_cache", "06_ln_qkv_rope_kv", "past_key_values.update", model_faithful=False
    ),
    LnAttnBisectStageSpec(
        "eager_attn",
        "07_ln_qkv_rope_kv_eager",
        "eager_attention_forward",
        model_faithful=False,
    ),
    LnAttnBisectStageSpec(
        "o_proj", "08_ln_decomposed_attn", "reshape + o_proj", model_faithful=False
    ),
    LnAttnBisectStageSpec(
        "native_self_attn",
        "09_ln_native_self_attn",
        "native self_attn, single output (NOT 0.77 graph — use LnAttnProfile)",
        model_faithful=False,
    ),
)

# Canonical repro: stacked outputs — see collect_fused_ln_attn_stacked_metrics().
LAYER0_LN_ATTN_FUSED_STAGE_NAMES: tuple[str, ...] = (
    "decode_setup",
    "input_layernorm",
    "self_attn",
)

LAYER0_LN_ATTN_MODEL_FAITHFUL_STAGES: tuple[LnAttnBisectStageSpec, ...] = (
    LnAttnBisectStageSpec(
        "layernorm", "01_layernorm", "input_layernorm (isolated)", model_faithful=True
    ),
    LnAttnBisectStageSpec(
        "fused_ln_attn",
        "02_ln_attn_fused_stack",
        "LnAttnProfile torch.stack (canonical ~0.77 graph)",
        model_faithful=True,
    ),
)

LAYER0_LN_ATTN_BISECT_STAGE_IDS: tuple[str, ...] = tuple(
    stage.stage_id for stage in LAYER0_LN_ATTN_BISECT_STAGES
)


def _anchor_output(primary: torch.Tensor, *prior: torch.Tensor) -> torch.Tensor:
    """Return ``primary`` but keep prior tensor ops in the autograd/FX graph."""
    if not prior:
        return primary
    anchor = primary
    for tensor in prior:
        anchor = anchor + tensor.sum() * 0.0
    return anchor


def layer0_ln_attn_bisect_forward(
    stop_after: str,
    inputs_embeds: torch.Tensor,
    past_key_values: Any,
    *,
    config: Any,
    rotary_emb: nn.Module,
    input_layernorm: nn.Module,
    self_attn: nn.Module,
) -> torch.Tensor:
    """
    Cumulative LN + attention partial forward; stops after ``stop_after`` stage id.

    Returns the **checkpoint tensor for that stage** (same tensor CPU golden uses).
    """
    if stop_after not in LAYER0_LN_ATTN_BISECT_STAGE_IDS:
        raise ValueError(
            f"Unknown stop_after {stop_after!r}; expected one of {LAYER0_LN_ATTN_BISECT_STAGE_IDS}"
        )

    past_key_values = prepare_decode_past_key_values(
        past_key_values, inputs_embeds.device
    )
    (
        hidden_states,
        past_key_values,
        causal_mask,
        position_embeddings,
        position_ids,
        cache_position,
        use_cache,
    ) = _llama_decode_step_tensors(
        inputs_embeds,
        past_key_values,
        config=config,
        rotary_emb=rotary_emb,
    )

    normed = input_layernorm(hidden_states)
    if stop_after == "layernorm":
        return normed

    if stop_after == "native_self_attn":
        attn_output, _ = self_attn(
            hidden_states=normed,
            attention_mask=causal_mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        return attn_output

    from transformers.models.llama.modeling_llama import (
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    input_shape = normed.shape[:-1]
    hidden_shape = (*input_shape, -1, self_attn.head_dim)

    query_states = self_attn.q_proj(normed).view(hidden_shape).transpose(1, 2)
    if stop_after == "q_proj":
        return query_states

    key_states = self_attn.k_proj(normed).view(hidden_shape).transpose(1, 2)
    if stop_after == "k_proj":
        return _anchor_output(key_states, query_states)

    value_states = self_attn.v_proj(normed).view(hidden_shape).transpose(1, 2)
    if stop_after == "v_proj":
        return _anchor_output(value_states, query_states, key_states)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    if stop_after == "rope":
        return _anchor_output(query_states, key_states, value_states)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self_attn.layer_idx, cache_kwargs
        )
    if stop_after == "kv_cache":
        return _anchor_output(key_states, query_states, value_states)

    attn_core, _ = eager_attention_forward(
        self_attn,
        query_states,
        key_states,
        value_states,
        causal_mask,
        dropout=0.0,
        scaling=self_attn.scaling,
    )
    if stop_after == "eager_attn":
        return _anchor_output(attn_core, query_states, key_states, value_states)

    attn_output = attn_core.reshape(*input_shape, -1).contiguous()
    attn_output = self_attn.o_proj(attn_output)
    if stop_after == "o_proj":
        return attn_output

    raise RuntimeError(f"Unhandled stop_after={stop_after!r}")


class JanusLlamaLayer0LnAttnBisect(nn.Module):
    """Incremental LN + attention partial graph for PCC bisection."""

    def __init__(
        self,
        llama_model: nn.Module,
        stop_after: str,
        layer_idx: int = DEFAULT_DECODE_LAYER_IDX,
    ):
        super().__init__()
        layer = llama_model.layers[layer_idx]
        self._stop_after = stop_after
        self._config = llama_model.config
        self.rotary_emb = llama_model.rotary_emb
        self.input_layernorm = layer.input_layernorm
        self.self_attn = layer.self_attn
        self.past_key_values = None

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        return layer0_ln_attn_bisect_forward(
            self._stop_after,
            inputs_embeds,
            self.past_key_values,
            config=self._config,
            rotary_emb=self.rotary_emb,
            input_layernorm=self.input_layernorm,
            self_attn=self.self_attn,
        )


def make_layer0_ln_attn_bisect_wrapper(
    llama_model: nn.Module,
    past_key_values: Any,
    stop_after: str,
) -> JanusLlamaLayer0LnAttnBisect:
    wrapper = JanusLlamaLayer0LnAttnBisect(llama_model, stop_after=stop_after)
    wrapper.past_key_values = past_key_values
    return wrapper


def collect_fused_ln_attn_stacked_metrics(
    llama_model: nn.Module,
    inputs_embeds: torch.Tensor,
    past_key_values: Any,
) -> dict[str, TensorMatchMetrics]:
    """
    Canonical ~0.77 graph: ``JanusLlamaDecoderLayer0LnAttnProfile`` (stacked outputs).

    Returns metrics for ``decode_setup``, ``input_layernorm``, and ``self_attn`` slices.
    """
    wrapper = JanusLlamaDecoderLayer0LnAttnProfile(llama_model)
    wrapper.past_key_values = past_key_values
    comparison_config = ComparisonConfig(assert_on_failure=False)
    evaluator = TorchComparisonEvaluator(comparison_config)
    rows: dict[str, TensorMatchMetrics] = {}

    def _comparator(
        device_stacked: torch.Tensor,
        cpu_stacked: torch.Tensor,
        _args: Any,
        _kwargs: Any,
    ) -> None:
        from tests.torch.models.janus_pro_pcc_drop.decoder_op_test_utils import (
            _tensor_match_metrics,
        )

        for index, name in enumerate(LAYER0_LN_ATTN_FUSED_STAGE_NAMES):
            rows[name] = _tensor_match_metrics(
                evaluator, device_stacked[index], cpu_stacked[index]
            )

    run_op_test(
        wrapper,
        [inputs_embeds],
        framework=Framework.TORCH,
        comparison_config=comparison_config,
        custom_comparator=_comparator,
    )
    return rows


def collect_layer0_ln_attn_bisect_metrics(
    llama_model: nn.Module,
    inputs_embeds: torch.Tensor,
    past_key_values: Any,
    stop_after: str,
) -> TensorMatchMetrics:
    wrapper = make_layer0_ln_attn_bisect_wrapper(
        llama_model, past_key_values, stop_after
    )
    return run_decoder_op_test_collect_metrics(wrapper, [inputs_embeds])


def print_layer0_ln_attn_bisect_profile(
    llama_model: nn.Module,
    inputs_embeds: torch.Tensor,
    past_key_values: Any,
    *,
    label: str = "layer0 ln→attn bisect profile",
    stages: tuple[LnAttnBisectStageSpec, ...] | None = None,
) -> list[tuple[LnAttnBisectStageSpec, TensorMatchMetrics]]:
    """Run incremental stages; print PCC table with ΔPCC vs previous stage."""
    if stages is None:
        stages = LAYER0_LN_ATTN_BISECT_STAGES

    rows: list[tuple[LnAttnBisectStageSpec, TensorMatchMetrics]] = []
    prev_pcc: float | None = None

    print(f"\n=== {label} ===")
    print(
        f"{'stage':<28}  {'graph':<10}  {'ops_added':<24}  {'pcc':>10}  {'Δpcc':>10}  "
        f"{'max_abs':>12}  {'rel_l2':>10}"
    )
    print("-" * 118)

    for stage in stages:
        metrics = collect_layer0_ln_attn_bisect_metrics(
            llama_model,
            inputs_embeds,
            past_key_values,
            stage.stage_id,
        )
        delta = metrics.pcc - prev_pcc if prev_pcc is not None else float("nan")
        delta_str = f"{delta:>10.6f}" if prev_pcc is not None else f"{'—':>10}"
        graph_tag = "model" if stage.model_faithful else "bisect"
        print(
            f"{stage.label:<28}  {graph_tag:<10}  {stage.ops_added:<24}  "
            f"{metrics.pcc:>10.6f}  {delta_str}  "
            f"{metrics.max_abs_diff:>12.6e}  {metrics.rel_l2_diff:>10.6e}"
        )
        rows.append((stage, metrics))
        prev_pcc = metrics.pcc

    print("-" * 118)
    print(
        "Bisect stages 02–09 use different TT graphs (single/decomposed outputs). "
        "They do NOT reproduce the ~0.77 ``self_attn`` bug. "
        "For that, use ``JanusLlamaDecoderLayer0LnAttnProfile`` (stacked outputs) or "
        "``test_image_token_decode_decoder_layer0_input_layernorm_self_attn_loader_pro_1b``."
    )
    return rows


def print_layer0_ln_attn_model_faithful_profile(
    llama_model: nn.Module,
    inputs_embeds: torch.Tensor,
    past_key_values: Any,
) -> None:
    """
    Canonical repro only: isolated LN + **stacked** LN+attn (``self_attn`` ~0.77).

    Does not use bisect single-output ``native_self_attn`` (~0.99 wrong graph).
    """
    print(f"\n=== Pro_1B layer0 LN+attn (canonical repro) ===")
    print(
        f"{'stage':<28}  {'graph':<10}  {'ops_added':<32}  {'pcc':>10}  "
        f"{'max_abs':>12}  {'rel_l2':>10}"
    )
    print("-" * 110)

    ln_metrics = collect_layer0_ln_attn_bisect_metrics(
        llama_model, inputs_embeds, past_key_values, "layernorm"
    )
    print(
        f"{'01_layernorm':<28}  {'isolated':<10}  {'input_layernorm only':<32}  "
        f"{ln_metrics.pcc:>10.6f}  {ln_metrics.max_abs_diff:>12.6e}  "
        f"{ln_metrics.rel_l2_diff:>10.6e}"
    )

    fused = collect_fused_ln_attn_stacked_metrics(
        llama_model, inputs_embeds, past_key_values
    )
    for name in LAYER0_LN_ATTN_FUSED_STAGE_NAMES:
        m = fused[name]
        label = f"02_fused_{name}" if name == "self_attn" else f"   fused_{name}"
        graph = "canonical" if name == "self_attn" else "canonical"
        print(
            f"{label:<28}  {graph:<10}  {name + ' (stacked graph)':<32}  "
            f"{m.pcc:>10.6f}  {m.max_abs_diff:>12.6e}  {m.rel_l2_diff:>10.6e}"
        )

    print("-" * 110)
    print(
        "Use ``self_attn`` row (~0.77) + ``JanusLlamaDecoderLayer0LnAttnProfile`` for "
        "pytest --serialize / tt-metal codegen. Bisect ladder does not reproduce this."
    )


def print_layer0_ln_attn_bisect_stage_metrics(
    stage: LnAttnBisectStageSpec,
    metrics: TensorMatchMetrics,
) -> None:
    print_tensor_match_metrics(
        f"{stage.label} ({stage.ops_added})",
        [("output", metrics)],
    )
