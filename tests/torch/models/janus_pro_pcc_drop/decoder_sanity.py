# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Wrappers and input helpers for ImageTokenStep decode decoder-layer op tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask

from third_party.tt_forge_models.janus_pro.text_to_image.pytorch.src.model_utils import (
    DTYPE,
    align_kv_cache_device,
    load_mmgpt,
    make_image_token_decode_inputs,
)


def load_image_token_decode_bundle(
    repo_id: str,
    dtype: Optional[torch.dtype] = None,
) -> dict[str, Any]:
    """
    Decode-step tensors for decoder-layer sanities.

    Same inputs as ``make_image_token_decode_inputs`` (CPU prefill KV + 1-token CFG embeds).
    """
    dtype = dtype if dtype is not None else DTYPE
    decode = make_image_token_decode_inputs(repo_id, dtype)
    mmgpt = load_mmgpt(repo_id, dtype)
    return {
        "mmgpt": mmgpt,
        "inputs_embeds": decode["inputs_embeds"],
        "past_key_values": decode["past_key_values"],
        "llama_model": mmgpt.language_model.model,
        "gen_head": mmgpt.gen_head,
    }


def _llama_decode_step_tensors(
    inputs_embeds: torch.Tensor,
    past_key_values,
    *,
    config: Any,
    rotary_emb: nn.Module,
) -> tuple[
    torch.Tensor,
    Any,
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    bool,
]:
    """Shared decode-step tensors matching ``LlamaModel.forward`` (prefill KV + decode embeds)."""
    use_cache = True
    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=config)

    past_seen_tokens = (
        past_key_values.get_seq_length() if past_key_values is not None else 0
    )
    cache_position = torch.arange(
        inputs_embeds.shape[1], device=inputs_embeds.device
    ) + past_seen_tokens
    position_ids = cache_position.unsqueeze(0)

    causal_mask = create_causal_mask(
        config=config,
        inputs_embeds=inputs_embeds,
        attention_mask=None,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    hidden_states = inputs_embeds
    position_embeddings = rotary_emb(hidden_states, position_ids=position_ids)
    return (
        hidden_states,
        past_key_values,
        causal_mask,
        position_embeddings,
        position_ids,
        cache_position,
        use_cache,
    )


def _llama_decode_step_tensors_from_model(
    llama_model: nn.Module,
    inputs_embeds: torch.Tensor,
    past_key_values,
) -> tuple[
    torch.Tensor,
    Any,
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    bool,
]:
    return _llama_decode_step_tensors(
        inputs_embeds,
        past_key_values,
        config=llama_model.config,
        rotary_emb=llama_model.rotary_emb,
    )


def _run_llama_decoder_layers(
    layers: nn.ModuleList,
    inputs_embeds: torch.Tensor,
    past_key_values,
    *,
    config: Any,
    rotary_emb: nn.Module,
    final_norm: nn.Module | None = None,
) -> torch.Tensor:
    """Run ``LlamaDecoderLayer`` loop + optional ``norm`` (same contract as ``LlamaModel``)."""
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

    for decoder_layer in layers:
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
        )

    if final_norm is not None:
        hidden_states = final_norm(hidden_states)
    return hidden_states


def llama_decode_hidden_states(
    llama_model: nn.Module,
    inputs_embeds: torch.Tensor,
    past_key_values,
    *,
    num_layers: Optional[int] = None,
    apply_final_norm: bool = True,
) -> torch.Tensor:
    """
    Replicate ``LlamaModel.forward`` for decode (inputs_embeds + KV cache).

    Matches the loop in transformers ``LlamaModel`` used by ImageTokenStep decode.
    """
    (
        hidden_states,
        past_key_values,
        causal_mask,
        position_embeddings,
        position_ids,
        cache_position,
        use_cache,
    ) = _llama_decode_step_tensors_from_model(
        llama_model, inputs_embeds, past_key_values
    )

    layer_limit = num_layers
    if layer_limit is None:
        layer_limit = llama_model.config.num_hidden_layers

    for decoder_layer in llama_model.layers[:layer_limit]:
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
        )

    if apply_final_norm and layer_limit == llama_model.config.num_hidden_layers:
        hidden_states = llama_model.norm(hidden_states)
    return hidden_states


def llama_decode_hidden_states_stacked_per_layer(
    llama_model: nn.Module,
    inputs_embeds: torch.Tensor,
    past_key_values,
) -> torch.Tensor:
    """
    Hidden state after each ``LlamaDecoderLayer`` (pre-``norm``).

    Returns shape ``[num_hidden_layers, batch, seq, hidden]`` for per-layer PCC profiling.
    """
    (
        hidden_states,
        past_key_values,
        causal_mask,
        position_embeddings,
        position_ids,
        cache_position,
        use_cache,
    ) = _llama_decode_step_tensors_from_model(
        llama_model, inputs_embeds, past_key_values
    )

    per_layer: list[torch.Tensor] = []
    for decoder_layer in llama_model.layers:
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        per_layer.append(hidden_states)
    return torch.stack(per_layer, dim=0)


@dataclass(frozen=True)
class DecoderLayerIsolatedContext:
    """Inputs to run exactly one ``LlamaDecoderLayer`` in decode (layer ``layer_idx``)."""

    layer_idx: int
    hidden_in: torch.Tensor
    past_key_values: Any
    causal_mask: torch.Tensor
    position_embeddings: tuple[torch.Tensor, torch.Tensor]
    position_ids: torch.Tensor
    cache_position: torch.Tensor
    use_cache: bool


@torch.inference_mode()
def compute_decoder_layer_isolated_context(
    llama_model: nn.Module,
    inputs_embeds: torch.Tensor,
    past_key_values: Any,
    layer_idx: int,
) -> DecoderLayerIsolatedContext:
    """
    Run layers ``0 .. layer_idx-1`` on CPU, snapshot hidden + KV before layer ``layer_idx``.
    """
    from tests.torch.models.janus_pro_pcc_drop.decoder_submodule_sanity import (
        clone_dynamic_cache,
    )

    kv = clone_dynamic_cache(past_key_values)
    (
        hidden_states,
        past_key_values,
        causal_mask,
        position_embeddings,
        position_ids,
        cache_position,
        use_cache,
    ) = _llama_decode_step_tensors_from_model(
        llama_model, inputs_embeds, clone_dynamic_cache(kv)
    )

    for idx in range(layer_idx):
        hidden_states = llama_model.layers[idx](
            hidden_states,
            attention_mask=causal_mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
        )

    return DecoderLayerIsolatedContext(
        layer_idx=layer_idx,
        hidden_in=hidden_states.detach().cpu().clone(),
        past_key_values=clone_dynamic_cache(past_key_values),
        causal_mask=causal_mask.detach().cpu().clone(),
        position_embeddings=(
            position_embeddings[0].detach().cpu().clone(),
            position_embeddings[1].detach().cpu().clone(),
        ),
        position_ids=position_ids.detach().cpu().clone(),
        cache_position=cache_position.detach().cpu().clone(),
        use_cache=use_cache,
    )


class JanusLlamaDecoderSingleLayerIsolated(nn.Module):
    """One ``LlamaDecoderLayer`` with decode context captured from CPU (layer ``layer_idx``)."""

    def __init__(self, decoder_layer: nn.Module, ctx: DecoderLayerIsolatedContext):
        super().__init__()
        self.decoder_layer = decoder_layer
        self._ctx = ctx

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from tests.torch.models.janus_pro_pcc_drop.decoder_submodule_sanity import (
            clone_dynamic_cache,
        )

        ctx = self._ctx
        device = hidden_states.device
        past_key_values = align_kv_cache_device(
            clone_dynamic_cache(ctx.past_key_values), device
        )
        return self.decoder_layer(
            hidden_states,
            attention_mask=ctx.causal_mask.to(device),
            position_embeddings=(
                ctx.position_embeddings[0].to(device),
                ctx.position_embeddings[1].to(device),
            ),
            position_ids=ctx.position_ids.to(device),
            past_key_values=past_key_values,
            use_cache=ctx.use_cache,
            cache_position=ctx.cache_position.to(device),
        )


class JanusLlamaDecoderDecodeLoop(nn.Module):
    """
    ``language_model.model`` decode path for ImageTokenStep (KV on module).

    Same call as ``JanusGitImageTokenStep`` without ``gen_head``.
    """

    def __init__(self, llama_model: nn.Module):
        super().__init__()
        self.llama_model = llama_model
        self.past_key_values = None

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        past_key_values = align_kv_cache_device(
            self.past_key_values, inputs_embeds.device
        )
        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=past_key_values,
        )
        return outputs.last_hidden_state


class JanusLlamaDecoderLayersLoop(nn.Module):
    """
    Decode forward matching ``LlamaModel`` for the first ``num_layers`` blocks.

    Only registers ``rotary_emb``, the selected ``LlamaDecoderLayer`` modules, and
    optionally ``norm`` (same subgraph as full ``language_model.model`` decode, without
    embedding the unused deeper layers in the TT compile).
    """

    def __init__(self, llama_model: nn.Module, num_layers: Optional[int] = None):
        super().__init__()
        self._config = llama_model.config
        self.rotary_emb = llama_model.rotary_emb
        total_layers = llama_model.config.num_hidden_layers
        layer_count = num_layers if num_layers is not None else total_layers
        self.layers = nn.ModuleList(list(llama_model.layers[:layer_count]))
        self.norm = llama_model.norm if layer_count == total_layers else None
        self.past_key_values = None

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        past_key_values = align_kv_cache_device(
            self.past_key_values, inputs_embeds.device
        )
        return _run_llama_decoder_layers(
            self.layers,
            inputs_embeds,
            past_key_values,
            config=self._config,
            rotary_emb=self.rotary_emb,
            final_norm=self.norm,
        )


class JanusLlamaDecoderNativeLayerDecode(JanusLlamaDecoderLayersLoop):
    """First ``LlamaDecoderLayer`` only — same decode contract as ``LlamaModel.forward``."""

    def __init__(self, llama_model: nn.Module):
        super().__init__(llama_model, num_layers=1)


class JanusLlamaDecoderLayersPCCProfile(nn.Module):
    """
    Decode layer loop returning stacked hidden states after every decoder layer.

    One TT compile compares PCC at depths 1..``num_hidden_layers`` without separate tests.
    """

    def __init__(self, llama_model: nn.Module):
        super().__init__()
        self.llama_model = llama_model
        self.past_key_values = None

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        past_key_values = align_kv_cache_device(
            self.past_key_values, inputs_embeds.device
        )
        return llama_decode_hidden_states_stacked_per_layer(
            self.llama_model,
            inputs_embeds,
            past_key_values,
        )


class JanusGenHeadDecode(nn.Module):
    """gen_head on last-token hidden state (ImageTokenStep decode tail)."""

    def __init__(self, gen_head: nn.Module):
        super().__init__()
        self.gen_head = gen_head

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.gen_head(hidden_states[:, -1, :])
