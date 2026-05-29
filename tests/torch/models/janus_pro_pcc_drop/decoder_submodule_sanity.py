# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Capture exact decode-step tensors and wrappers for isolated Llama submodule op tests."""

from __future__ import annotations

import copy
import inspect
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn

from tests.torch.models.janus_pro_pcc_drop.decoder_sanity import (
    _llama_decode_step_tensors,
    _llama_decode_step_tensors_from_model,
    llama_decode_hidden_states,
    load_image_token_decode_bundle,
)
from third_party.tt_forge_models.janus_pro.text_to_image.pytorch.src.model_utils import (
    align_kv_cache_device,
)


# Layer 0 matches the first failing depth in the decode PCC profile (issue #4968).
DEFAULT_DECODE_LAYER_IDX = 0


def clone_dynamic_cache(cache: Any) -> Any:
    """Independent KV cache for CPU/TT op tests (attention mutates cache in-place)."""
    if cache is None:
        return None
    return copy.deepcopy(cache)


def prepare_decode_past_key_values(cache: Any, device: torch.device) -> Any:
    """
    Fresh KV snapshot for one ``forward`` call.

    Use for **isolated** op tests / bisect (CPU then TT on same module).

    Do **not** use in ``JanusLlamaDecoderLayer0LnAttnProfile`` / stacked LN+attn repro —
    cloning inside that ``forward`` changes TT fusion and PCC jumps from ~0.77 to ~0.99
    (see ``janus_logs_2/layer0_ln_attn_loader.log`` vs ``final_reproducible_sanity.log``).
    """
    if cache is None:
        return None
    return align_kv_cache_device(clone_dynamic_cache(cache), device)


@dataclass
class DecoderSubmoduleFixtures:
    """Exact tensors and modules from one ImageTokenStep decode forward."""

    layer_idx: int
    rotary_emb: nn.Module
    input_layernorm: nn.Module
    self_attn: nn.Module
    post_attention_layernorm: nn.Module
    mlp: nn.Module
    final_norm: nn.Module
    rotary_hidden_states: torch.Tensor
    rotary_position_ids: torch.Tensor
    input_layernorm_hidden: torch.Tensor
    attn_hidden_states: torch.Tensor
    attn_attention_mask: torch.Tensor
    attn_position_embeddings: tuple[torch.Tensor, torch.Tensor]
    attn_cache_position: torch.Tensor
    attn_kv_snapshot: Any
    post_attention_layernorm_hidden: torch.Tensor
    mlp_hidden: torch.Tensor
    final_norm_hidden: torch.Tensor


@torch.inference_mode()
def capture_decoder_submodule_fixtures(
    llama_model: nn.Module,
    inputs_embeds: torch.Tensor,
    past_key_values: Any,
    *,
    layer_idx: int = DEFAULT_DECODE_LAYER_IDX,
) -> DecoderSubmoduleFixtures:
    """
    Run decode setup and layer ``layer_idx`` step-by-step; record submodule I/O.

    ``past_key_values`` is the CPU cache from ``make_image_token_decode_inputs`` (prefill).
    """
    kv_initial = clone_dynamic_cache(past_key_values)

    (
        hidden_states,
        past_key_values,
        causal_mask,
        position_embeddings,
        position_ids,
        cache_position,
        _use_cache,
    ) = _llama_decode_step_tensors_from_model(
        llama_model, inputs_embeds, clone_dynamic_cache(kv_initial)
    )

    layer = llama_model.layers[layer_idx]

    rotary_hidden = hidden_states
    input_ln_hidden = hidden_states.clone()
    normed_for_attn = layer.input_layernorm(hidden_states)

    kv_before_attn = clone_dynamic_cache(past_key_values)
    attn_hidden = normed_for_attn.clone()

    attn_out, _ = layer.self_attn(
        hidden_states=normed_for_attn,
        attention_mask=causal_mask,
        position_embeddings=position_embeddings,
        past_key_values=past_key_values,
        cache_position=cache_position,
    )
    hidden_after_attn = input_ln_hidden + attn_out
    post_attn_ln_hidden = hidden_after_attn.clone()
    mlp_hidden = layer.post_attention_layernorm(hidden_after_attn).clone()

    final_hidden = llama_decode_hidden_states(
        llama_model,
        inputs_embeds,
        clone_dynamic_cache(kv_initial),
        apply_final_norm=False,
    )

    return DecoderSubmoduleFixtures(
        layer_idx=layer_idx,
        rotary_emb=llama_model.rotary_emb,
        input_layernorm=layer.input_layernorm,
        self_attn=layer.self_attn,
        post_attention_layernorm=layer.post_attention_layernorm,
        mlp=layer.mlp,
        final_norm=llama_model.norm,
        rotary_hidden_states=rotary_hidden,
        rotary_position_ids=position_ids,
        input_layernorm_hidden=input_ln_hidden,
        attn_hidden_states=attn_hidden,
        attn_attention_mask=causal_mask,
        attn_position_embeddings=(
            position_embeddings[0].clone(),
            position_embeddings[1].clone(),
        ),
        attn_cache_position=cache_position,
        attn_kv_snapshot=kv_before_attn,
        post_attention_layernorm_hidden=post_attn_ln_hidden,
        mlp_hidden=mlp_hidden,
        final_norm_hidden=final_hidden,
    )


def load_decoder_submodule_fixtures(
    repo_id: str,
    dtype: Optional[torch.dtype] = None,
    *,
    layer_idx: int = DEFAULT_DECODE_LAYER_IDX,
) -> DecoderSubmoduleFixtures:
    bundle = load_image_token_decode_bundle(repo_id, dtype=dtype)
    return capture_decoder_submodule_fixtures(
        bundle["llama_model"],
        bundle["inputs_embeds"],
        bundle["past_key_values"],
        layer_idx=layer_idx,
    )


class JanusLlamaRotaryEmbDecode(nn.Module):
    def __init__(self, rotary_emb: nn.Module):
        super().__init__()
        self.rotary_emb = rotary_emb

    def forward(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.rotary_emb(hidden_states, position_ids=position_ids)


class JanusLlamaRMSNormDecode(nn.Module):
    def __init__(self, norm: nn.Module):
        super().__init__()
        self.norm = norm

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)


class JanusLlamaMLPDecode(nn.Module):
    def __init__(self, mlp: nn.Module):
        super().__init__()
        self.mlp = mlp

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(hidden_states)


class JanusLlamaSelfAttnNativeDecode(nn.Module):
    """
    Layer-0 ``self_attn`` only — native ``LlamaAttention.forward`` (same as combined / full model).

    Does not use ``JanusLlamaAttentionDecode`` (that wrapper can hit TT SDPA-decode compile errors).
    Decode context (KV, RoPE, mask) comes from ``inputs_embeds`` + ``past_key_values`` on the module.
    Forward input is post-``input_layernorm`` hidden states only.
    """

    def __init__(self, llama_model: nn.Module, layer_idx: int = DEFAULT_DECODE_LAYER_IDX):
        super().__init__()
        self._config = llama_model.config
        self.rotary_emb = llama_model.rotary_emb
        self.self_attn = llama_model.layers[layer_idx].self_attn
        self.past_key_values = None
        self.inputs_embeds_for_decode_context: torch.Tensor | None = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.inputs_embeds_for_decode_context is None:
            raise RuntimeError(
                "Set wrapper.inputs_embeds_for_decode_context from decode bundle before run_op_test"
            )
        past_key_values = prepare_decode_past_key_values(
            self.past_key_values, hidden_states.device
        )
        inputs_embeds = self.inputs_embeds_for_decode_context.to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        (
            _hidden_states,
            past_key_values,
            causal_mask,
            position_embeddings,
            position_ids,
            cache_position,
            use_cache,
        ) = _llama_decode_step_tensors(
            inputs_embeds,
            past_key_values,
            config=self._config,
            rotary_emb=self.rotary_emb,
        )
        attn_output, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=causal_mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        return attn_output


def make_layer0_self_attn_native_wrapper(
    llama_model: nn.Module,
    decode_bundle: dict,
) -> JanusLlamaSelfAttnNativeDecode:
    """Native layer-0 attention op with real decode KV / mask / RoPE context."""
    wrapper = JanusLlamaSelfAttnNativeDecode(llama_model)
    wrapper.past_key_values = decode_bundle["past_key_values"]
    wrapper.inputs_embeds_for_decode_context = decode_bundle["inputs_embeds"]
    return wrapper


class JanusLlamaAttentionDecode(nn.Module):
    """``LlamaAttention`` with decode KV/mask/RoPE taken from a captured forward (legacy wrapper)."""

    def __init__(
        self,
        self_attn: nn.Module,
        *,
        attention_mask: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cache_position: torch.Tensor,
        kv_snapshot: Any,
    ):
        super().__init__()
        self.self_attn = self_attn
        # transformers 5.x may return ``None`` from ``create_causal_mask`` when using KV cache.
        if attention_mask is not None:
            self.register_buffer("_attention_mask", attention_mask.detach().cpu())
        else:
            self._attention_mask = None
        self.register_buffer("_cache_position", cache_position.detach().cpu())
        self.register_buffer("_cos", position_embeddings[0].detach().cpu())
        self.register_buffer("_sin", position_embeddings[1].detach().cpu())
        self._kv_snapshot = kv_snapshot

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        device = hidden_states.device
        past_key_values = align_kv_cache_device(
            clone_dynamic_cache(self._kv_snapshot), device
        )
        position_embeddings = (
            self._cos.to(device=device, dtype=hidden_states.dtype),
            self._sin.to(device=device, dtype=hidden_states.dtype),
        )
        attention_mask = None
        if self._attention_mask is not None:
            attention_mask = self._attention_mask.to(device=device)
        attn_output, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            cache_position=self._cache_position.to(device=device),
        )
        return attn_output


# Checkpoints after each step in ``JanusLlamaDecoderLayer0Combined`` (last = full layer out).
LAYER0_COMBINED_STAGE_NAMES: tuple[str, ...] = (
    "decode_setup",
    "input_layernorm",
    "self_attn",
    "post_attention_layernorm",
    "mlp",
    "layer_output",
)

# ``input_layernorm`` + ``self_attn`` only (reproduce fused-graph drop after layernorm).
LAYER0_LN_ATTN_STAGE_NAMES_FROM_EMBEDS: tuple[str, ...] = (
    "decode_setup",
    "input_layernorm",
    "self_attn",
)
LAYER0_LN_ATTN_STAGE_NAMES_FROM_HIDDEN: tuple[str, ...] = (
    "input_layernorm",
    "self_attn",
)

# Per-op checkpoints inside ``LlamaAttention.forward`` (CPU reference only).
LAYER0_LN_ATTN_ATTENTION_INTERNAL_STAGE_NAMES: tuple[str, ...] = (
    "attn_q_proj",
    "attn_k_proj",
    "attn_v_proj",
    "attn_post_rope",
    "attn_post_kv_cache",
    "attn_post_eager",
    "attn_o_proj",
)


def describe_llama_self_attn(self_attn: nn.Module) -> dict[str, Any]:
    """
    Where ``self_attn.forward`` lives and which attention ops run for Janus loads.

    Janus ``load_mmgpt`` sets ``attn_implementation='eager'`` (see ``model_utils.py``), so
    ``LlamaAttention.forward`` in HuggingFace ``modeling_llama.py`` calls ``eager_attention_forward``
    (QK^T, mask, softmax, @V) — not SDPA/flash unless config overrides.
    """
    from transformers.models.llama.modeling_llama import LlamaAttention

    config = self_attn.config
    return {
        "class": type(self_attn).__name__,
        "module": type(self_attn).__module__,
        "forward_file": inspect.getfile(type(self_attn)),
        "forward_qualname": f"{type(self_attn).__module__}.{type(self_attn).__name__}.forward",
        "is_llama_attention": isinstance(self_attn, LlamaAttention),
        "attn_implementation": getattr(config, "_attn_implementation", None),
        "layer_idx": getattr(self_attn, "layer_idx", None),
        "hidden_size": config.hidden_size,
        "num_heads": config.num_attention_heads,
        "num_kv_heads": config.num_key_value_heads,
        "head_dim": getattr(self_attn, "head_dim", None),
        "submodules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "forward_stages": [
            "q_proj + reshape/transpose",
            "k_proj + reshape/transpose",
            "v_proj + reshape/transpose",
            "apply_rotary_pos_emb (cos/sin from model rotary_emb)",
            "past_key_values.update (decode KV)",
            "eager_attention_forward (matmul, mask, softmax, matmul)",
            "reshape + o_proj",
        ],
    }


def _layer0_combined_forward_checkpoints(
    *,
    inputs_embeds: torch.Tensor,
    past_key_values: Any,
    config: Any,
    rotary_emb: nn.Module,
    input_layernorm: nn.Module,
    self_attn: nn.Module,
    post_attention_layernorm: nn.Module,
    mlp: nn.Module,
) -> list[torch.Tensor]:
    """Shared layer-0 combined forward; one tensor per stage for PCC profiling."""
    past_key_values = align_kv_cache_device(past_key_values, inputs_embeds.device)
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

    checkpoints: list[torch.Tensor] = [hidden_states]

    residual = hidden_states
    hidden_states = input_layernorm(hidden_states)
    checkpoints.append(hidden_states)

    attn_output, _ = self_attn(
        hidden_states=hidden_states,
        attention_mask=causal_mask,
        position_embeddings=position_embeddings,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        cache_position=cache_position,
    )
    checkpoints.append(attn_output)

    hidden_states = residual + attn_output
    residual = hidden_states
    hidden_states = post_attention_layernorm(hidden_states)
    checkpoints.append(hidden_states)

    hidden_states = mlp(hidden_states)
    checkpoints.append(hidden_states)

    checkpoints.append(residual + hidden_states)
    return checkpoints


def _layer0_decode_attn_context(
    inputs_embeds: torch.Tensor,
    past_key_values: Any,
    *,
    config: Any,
    rotary_emb: nn.Module,
) -> tuple[Any, torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, bool]:
    """Mask / RoPE / KV for layer-0 ``self_attn`` (shared by ln+attn profiles)."""
    past_key_values = align_kv_cache_device(past_key_values, inputs_embeds.device)
    (
        _hidden_states,
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
    return (
        past_key_values,
        causal_mask,
        position_embeddings,
        position_ids,
        cache_position,
        use_cache,
    )


def _layer0_ln_attn_checkpoints_from_embeds(
    inputs_embeds: torch.Tensor,
    past_key_values: Any,
    *,
    config: Any,
    rotary_emb: nn.Module,
    input_layernorm: nn.Module,
    self_attn: nn.Module,
) -> list[torch.Tensor]:
    """decode_setup → ``input_layernorm`` → ``self_attn`` (attn output tensor)."""
    # align only — do NOT clone here (preserves canonical ~0.77 fused LN+attn TT graph).
    past_key_values = align_kv_cache_device(past_key_values, inputs_embeds.device)
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
    checkpoints: list[torch.Tensor] = [hidden_states]

    hidden_states = input_layernorm(hidden_states)
    checkpoints.append(hidden_states)

    attn_output, _ = self_attn(
        hidden_states=hidden_states,
        attention_mask=causal_mask,
        position_embeddings=position_embeddings,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        cache_position=cache_position,
    )
    checkpoints.append(attn_output)
    return checkpoints


def _layer0_ln_attn_checkpoints_from_hidden(
    hidden_before_input_layernorm: torch.Tensor,
    inputs_embeds: torch.Tensor,
    past_key_values: Any,
    *,
    config: Any,
    rotary_emb: nn.Module,
    input_layernorm: nn.Module,
    self_attn: nn.Module,
) -> list[torch.Tensor]:
    """Saved hidden before LN + decode context from embeds/KV (no decode_setup checkpoint)."""
    (
        past_key_values,
        causal_mask,
        position_embeddings,
        position_ids,
        cache_position,
        use_cache,
    ) = _layer0_decode_attn_context(
        inputs_embeds,
        past_key_values,
        config=config,
        rotary_emb=rotary_emb,
    )

    hidden_states = input_layernorm(hidden_before_input_layernorm)
    checkpoints: list[torch.Tensor] = [hidden_states]

    attn_output, _ = self_attn(
        hidden_states=hidden_states,
        attention_mask=causal_mask,
        position_embeddings=position_embeddings,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        cache_position=cache_position,
    )
    checkpoints.append(attn_output)
    return checkpoints


def _llama_self_attn_forward_checkpoints(
    self_attn: nn.Module,
    hidden_states: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    past_key_values: Any,
    cache_position: torch.Tensor,
) -> list[torch.Tensor]:
    """
    Explicit ``LlamaAttention.forward`` stages (matches HF ``modeling_llama.py`` eager path).

    **CPU reference only** — uses ``clone_dynamic_cache`` and must not run inside a
    Dynamo-traced ``nn.Module.forward`` (breaks TT compile).
    """
    from transformers.models.llama.modeling_llama import (
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    past_key_values = align_kv_cache_device(
        clone_dynamic_cache(past_key_values), hidden_states.device
    )

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self_attn.head_dim)

    query_states = self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    query_after_q_proj = query_states
    key_after_k_proj = key_states
    value_after_v_proj = value_states

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    query_after_rope = query_states

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self_attn.layer_idx, cache_kwargs
        )
    key_after_kv_cache = key_states

    attn_core, _ = eager_attention_forward(
        self_attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0,
        scaling=self_attn.scaling,
    )

    attn_output = attn_core.reshape(*input_shape, -1).contiguous()
    attn_output = self_attn.o_proj(attn_output)

    return [
        query_after_q_proj,
        key_after_k_proj,
        value_after_v_proj,
        query_after_rope,
        key_after_kv_cache,
        attn_core,
        attn_output,
    ]


@torch.inference_mode()
def cpu_reference_layer0_ln_attn_stages(
    llama_model: nn.Module,
    inputs_embeds: torch.Tensor,
    past_key_values: Any,
    *,
    layer_idx: int = DEFAULT_DECODE_LAYER_IDX,
) -> dict[str, torch.Tensor]:
    """
    CPU-only stage tensors for layer-0 ``input_layernorm`` + ``self_attn``.

    Uses the same ops as ``JanusLlamaDecoderLayer0LnAttnProfile`` for the fused path,
    plus an explicit eager decomposition of ``LlamaAttention`` for reference (not TT traced).
    """
    layer = llama_model.layers[layer_idx]
    past_key_values = align_kv_cache_device(past_key_values, inputs_embeds.device)
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
        config=llama_model.config,
        rotary_emb=llama_model.rotary_emb,
    )

    stages: dict[str, torch.Tensor] = {
        "decode_setup": hidden_states.detach().cpu(),
    }

    normed_hidden = layer.input_layernorm(hidden_states)
    stages["input_layernorm"] = normed_hidden.detach().cpu()

    kv_for_fused = clone_dynamic_cache(past_key_values)
    attn_out_fused, _ = layer.self_attn(
        hidden_states=normed_hidden,
        attention_mask=causal_mask,
        position_embeddings=position_embeddings,
        position_ids=position_ids,
        past_key_values=kv_for_fused,
        use_cache=use_cache,
        cache_position=cache_position,
    )
    stages["self_attn_fused"] = attn_out_fused.detach().cpu()

    internal = _llama_self_attn_forward_checkpoints(
        layer.self_attn,
        normed_hidden,
        attention_mask=causal_mask,
        position_embeddings=position_embeddings,
        past_key_values=past_key_values,
        cache_position=cache_position,
    )
    for name, tensor in zip(LAYER0_LN_ATTN_ATTENTION_INTERNAL_STAGE_NAMES, internal):
        stages[name] = tensor.detach().cpu()

    return stages


def print_cpu_layer0_ln_attn_reference(
    llama_model: nn.Module,
    inputs_embeds: torch.Tensor,
    past_key_values: Any,
    *,
    label: str = "CPU layer0 ln+attn reference",
) -> dict[str, torch.Tensor]:
    """Print CPU stage stats; verify fused ``self_attn`` matches manual ``attn_o_proj``."""
    stages = cpu_reference_layer0_ln_attn_stages(
        llama_model, inputs_embeds, past_key_values
    )
    fused = stages["self_attn_fused"].float()
    manual = stages["attn_o_proj"].float()
    max_abs = (fused - manual).abs().max().item()
    print(f"\n--- {label} ---")
    print(f"CPU sanity: fused self_attn vs manual attn_o_proj max_abs = {max_abs:.6e}")
    print(f"{'stage':<24}  {'shape':<22}  {'mean':>12}  {'std':>12}  {'maxabs':>12}")
    print("-" * 88)
    for name, tensor in stages.items():
        values = tensor.float()
        print(
            f"{name:<24}  {str(tuple(tensor.shape)):<22}  "
            f"{values.mean().item():>12.6e}  {values.std().item():>12.6e}  "
            f"{values.abs().max().item():>12.6e}"
        )
    print("-" * 88)
    return stages


class JanusLlamaDecoderLayer0LnAttnProfile(nn.Module):
    """
    Layer-0 ``input_layernorm`` + native ``self_attn`` in one TT op graph.

    Same subgraph as ``LlamaDecoderLayer.forward`` through attention (no manual q/k/v split).
    Matches the failing ~0.77 PCC when LN and attn are fused on TT.

    ``entry="embeds"``: forward ``inputs_embeds`` (loader decode bundle).
    ``entry="hidden"``: forward ``hidden_before_input_layernorm``; decode context from
    ``inputs_embeds_for_decode_context`` + ``past_key_values`` on the module.
    """

    def __init__(
        self,
        llama_model: nn.Module,
        *,
        entry: str = "embeds",
        layer_idx: int = DEFAULT_DECODE_LAYER_IDX,
    ):
        super().__init__()
        if entry not in ("embeds", "hidden"):
            raise ValueError(f"entry must be 'embeds' or 'hidden', got {entry!r}")
        self._entry = entry
        layer = llama_model.layers[layer_idx]
        self._config = llama_model.config
        self.rotary_emb = llama_model.rotary_emb
        self.input_layernorm = layer.input_layernorm
        self.self_attn = layer.self_attn
        self.past_key_values = None
        self.inputs_embeds_for_decode_context: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._entry == "embeds":
            checkpoints = _layer0_ln_attn_checkpoints_from_embeds(
                x,
                self.past_key_values,
                config=self._config,
                rotary_emb=self.rotary_emb,
                input_layernorm=self.input_layernorm,
                self_attn=self.self_attn,
            )
        else:
            if self.inputs_embeds_for_decode_context is None:
                raise RuntimeError(
                    "Set inputs_embeds_for_decode_context when entry='hidden'"
                )
            inputs_embeds = self.inputs_embeds_for_decode_context.to(
                device=x.device, dtype=x.dtype
            )
            checkpoints = _layer0_ln_attn_checkpoints_from_hidden(
                x,
                inputs_embeds,
                self.past_key_values,
                config=self._config,
                rotary_emb=self.rotary_emb,
                input_layernorm=self.input_layernorm,
                self_attn=self.self_attn,
            )
        return torch.stack(checkpoints, dim=0)


class JanusLlamaDecoderLayer0Combined(nn.Module):
    """
    Layer 0 as explicit submodule calls — same modules and flow as ``LlamaDecoderLayer.forward``.

    Matches ``JanusLlamaDecoderNativeLayerDecode`` (expect the same ~0.978 decode PCC on layer 0).

    Isolated op tests may wrap submodules (e.g. ``JanusLlamaAttentionDecode``) for single-op
    compile; that wrapper must not be used here — it changes the fused TT graph and triggers
    SDPA-decode compile failures unrelated to full-model layer 0.
    """

    def __init__(self, llama_model: nn.Module, layer_idx: int = DEFAULT_DECODE_LAYER_IDX):
        super().__init__()
        layer = llama_model.layers[layer_idx]
        self._config = llama_model.config
        self.rotary_emb = llama_model.rotary_emb
        self.input_layernorm = layer.input_layernorm
        self.self_attn = layer.self_attn
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.mlp = layer.mlp
        self.past_key_values = None

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        checkpoints = _layer0_combined_forward_checkpoints(
            inputs_embeds=inputs_embeds,
            past_key_values=self.past_key_values,
            config=self._config,
            rotary_emb=self.rotary_emb,
            input_layernorm=self.input_layernorm,
            self_attn=self.self_attn,
            post_attention_layernorm=self.post_attention_layernorm,
            mlp=self.mlp,
        )
        return checkpoints[-1]


class JanusLlamaDecoderLayer0CombinedProfile(nn.Module):
    """Layer 0 combined forward with stacked hidden states per submodule stage."""

    def __init__(self, llama_model: nn.Module, layer_idx: int = DEFAULT_DECODE_LAYER_IDX):
        super().__init__()
        layer = llama_model.layers[layer_idx]
        self._config = llama_model.config
        self.rotary_emb = llama_model.rotary_emb
        self.input_layernorm = layer.input_layernorm
        self.self_attn = layer.self_attn
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.mlp = layer.mlp
        self.past_key_values = None

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        checkpoints = _layer0_combined_forward_checkpoints(
            inputs_embeds=inputs_embeds,
            past_key_values=self.past_key_values,
            config=self._config,
            rotary_emb=self.rotary_emb,
            input_layernorm=self.input_layernorm,
            self_attn=self.self_attn,
            post_attention_layernorm=self.post_attention_layernorm,
            mlp=self.mlp,
        )
        return torch.stack(checkpoints, dim=0)
