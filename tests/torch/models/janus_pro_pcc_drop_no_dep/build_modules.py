# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Build HF Llama layer-0 modules from ``Layer0ModuleSpec`` (random init, no Janus load)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask

from tests.torch.models.janus_pro_pcc_drop.decoder_submodule_sanity import (
    _layer0_ln_attn_checkpoints_from_embeds,
)
from tests.torch.models.janus_pro_pcc_drop_no_dep.arch_specs import Layer0ModuleSpec
from tests.torch.models.janus_pro_pcc_drop_no_dep.constants import DTYPE
from tests.torch.models.janus_pro_pcc_drop_no_dep.kv_cache import align_kv_cache_device
from tests.torch.models.janus_pro_pcc_drop_no_dep.saved_fixtures import (
    apply_saved_layer0_weights,
    load_saved_decode_inputs,
    saved_fixtures_available,
)
from transformers import LlamaConfig


def _build_hf_modules(
    spec: Layer0ModuleSpec,
    *,
    dtype: torch.dtype,
) -> tuple[nn.Module, nn.Module, nn.Module]:
    from transformers.models.llama.modeling_llama import (
        LlamaAttention,
        LlamaRotaryEmbedding,
        LlamaRMSNorm,
    )

    config = spec.to_llama_config()
    rotary_emb = LlamaRotaryEmbedding(config=config)
    input_layernorm = LlamaRMSNorm(spec.hidden_size, eps=spec.rms_norm_eps)
    self_attn = LlamaAttention(config, layer_idx=0)

    for module in (rotary_emb, input_layernorm, self_attn):
        module.to(dtype=dtype)
        module.eval()

    _assert_linear_shapes(spec, input_layernorm, self_attn)
    return rotary_emb, input_layernorm, self_attn


def _assert_linear_shapes(
    spec: Layer0ModuleSpec,
    input_layernorm: nn.Module,
    self_attn: nn.Module,
) -> None:
    h = spec.hidden_size
    assert tuple(input_layernorm.weight.shape) == (h,)
    for name, expected in (
        ("q_proj", spec.q_proj),
        ("k_proj", spec.k_proj),
        ("v_proj", spec.v_proj),
        ("o_proj", spec.o_proj),
    ):
        linear = getattr(self_attn, name)
        got = (linear.in_features, linear.out_features)
        assert got == expected, f"{name}: expected Linear{expected}, got Linear{got}"


@torch.inference_mode()
def make_random_decode_inputs(
    spec: Layer0ModuleSpec,
    rotary_emb: nn.Module,
    self_attn: nn.Module,
    *,
    dtype: torch.dtype = DTYPE,
    seed: int = 42,
) -> tuple[torch.Tensor, Any]:
    """Random 1-token decode embeds + KV from short CPU prefill (same shapes as forge decode)."""
    torch.manual_seed(seed)
    device = torch.device("cpu")
    config = spec.to_llama_config()
    batch = spec.decode_batch
    prefill_len = spec.prefill_len

    prefill_hidden = (
        torch.randn(batch, prefill_len, spec.hidden_size, dtype=dtype, device=device) * 0.02
    )
    past_key_values = DynamicCache(config=config)
    cache_position = torch.arange(prefill_len, device=device)
    position_ids = cache_position.unsqueeze(0).expand(batch, -1)
    position_embeddings = rotary_emb(prefill_hidden, position_ids=position_ids)
    causal_mask = create_causal_mask(
        config=config,
        inputs_embeds=prefill_hidden,
        attention_mask=None,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )
    self_attn(
        hidden_states=prefill_hidden,
        attention_mask=causal_mask,
        position_embeddings=position_embeddings,
        past_key_values=past_key_values,
        use_cache=True,
        cache_position=cache_position,
    )

    decode_embeds = (
        torch.randn(
            batch,
            spec.decode_seq_len,
            spec.hidden_size,
            dtype=dtype,
            device=device,
        )
        * 0.02
    )
    return decode_embeds, past_key_values


@dataclass
class Layer0NoDepBundle:
    spec: Layer0ModuleSpec
    rotary_emb: nn.Module
    input_layernorm: nn.Module
    self_attn: nn.Module
    inputs_embeds: torch.Tensor
    past_key_values: Any
    llama_config: Any | None = None


def build_layer0_no_dep(
    spec: Layer0ModuleSpec,
    *,
    dtype: torch.dtype = DTYPE,
    seed: int = 42,
    use_saved_inputs: bool = False,
    load_hf_weights: bool = False,
) -> Layer0NoDepBundle:
    rotary_emb, input_layernorm, self_attn = _build_hf_modules(spec, dtype=dtype)
    if use_saved_inputs:
        if not saved_fixtures_available(spec.variant):
            raise FileNotFoundError(
                f"Saved decode fixtures missing for {spec.variant}; run test_save_layer0_no_dep_fixtures first."
            )
        inputs_embeds, past_key_values = load_saved_decode_inputs(spec.variant)
        inputs_embeds = inputs_embeds.to(dtype=dtype)
    else:
        inputs_embeds, past_key_values = make_random_decode_inputs(
            spec, rotary_emb, self_attn, dtype=dtype, seed=seed
        )
    bundle = Layer0NoDepBundle(
        spec=spec,
        rotary_emb=rotary_emb,
        input_layernorm=input_layernorm,
        self_attn=self_attn,
        inputs_embeds=inputs_embeds,
        past_key_values=past_key_values,
    )
    if use_saved_inputs and saved_fixtures_available(spec.variant):
        apply_saved_layer0_weights(bundle, spec.variant, dtype=dtype)
    elif load_hf_weights or os.environ.get("JANUS_NO_DEP_HF_WEIGHTS", "0") == "1":
        load_hf_layer0_weights(bundle, dtype=dtype)
    return bundle


class Layer0LnAttnNoDep(nn.Module):
    """
    Canonical stacked graph: decode_setup → input_layernorm → self_attn.

    Same sequence as ``JanusLlamaDecoderLayer0LnAttnProfile`` (``entry="embeds"``).
  """

    def __init__(self, bundle: Layer0NoDepBundle):
        super().__init__()
        self._config = bundle.llama_config or bundle.spec.to_llama_config()
        self.rotary_emb = bundle.rotary_emb
        self.input_layernorm = bundle.input_layernorm
        self.self_attn = bundle.self_attn
        self.past_key_values = bundle.past_key_values
        self._inputs_embeds = bundle.inputs_embeds

    @classmethod
    def from_spec(cls, spec: Layer0ModuleSpec, **kwargs: Any) -> Layer0LnAttnNoDep:
        return cls(build_layer0_no_dep(spec, **kwargs))

    @property
    def inputs_embeds_decode(self) -> torch.Tensor:
        return self._inputs_embeds

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        past_key_values = align_kv_cache_device(
            self.past_key_values, inputs_embeds.device
        )
        checkpoints = _layer0_ln_attn_checkpoints_from_embeds(
            inputs_embeds,
            past_key_values,
            config=self._config,
            rotary_emb=self.rotary_emb,
            input_layernorm=self.input_layernorm,
            self_attn=self.self_attn,
        )
        return torch.stack(checkpoints, dim=0)


def load_hf_layer0_weights(
    bundle: Layer0NoDepBundle,
    *,
    dtype: torch.dtype = DTYPE,
) -> None:
    """Copy layer-0 ``rotary_emb`` / ``input_layernorm`` / ``self_attn`` from Janus-Pro checkpoint."""
    import inspect

    from tests.runner.requirements import RequirementsManager

    import third_party.tt_forge_models.janus_pro.text_to_image.pytorch.loader as janus_loader
    from third_party.tt_forge_models.janus_pro.text_to_image.pytorch import (
        ModelLoader,
        ModelVariant,
    )
    from third_party.tt_forge_models.janus_pro.text_to_image.pytorch.src.model_utils import (
        load_mmgpt,
    )

    variant = ModelVariant(bundle.spec.variant)
    loader_path = inspect.getsourcefile(janus_loader)
    with RequirementsManager.for_loader(loader_path, framework="torch"):
        mmgpt = load_mmgpt(ModelLoader(variant)._repo_id(), dtype)
        llama = mmgpt.language_model.model
        layer = llama.layers[0]
        bundle.llama_config = llama.config
        bundle.rotary_emb.load_state_dict(llama.rotary_emb.state_dict(), strict=True)
        bundle.input_layernorm.load_state_dict(
            layer.input_layernorm.state_dict(), strict=True
        )
        bundle.self_attn.load_state_dict(layer.self_attn.state_dict(), strict=True)
    for module in (bundle.rotary_emb, bundle.input_layernorm, bundle.self_attn):
        module.to(dtype=dtype)
        module.eval()


def load_hf_layer0_weights_if_requested(
    bundle: Layer0NoDepBundle,
    *,
    dtype: torch.dtype = DTYPE,
) -> None:
    """Optional: ``JANUS_NO_DEP_HF_WEIGHTS=1`` copies weights from Janus-Pro checkpoint."""
    if os.environ.get("JANUS_NO_DEP_HF_WEIGHTS", "0") != "1":
        return
    load_hf_layer0_weights(bundle, dtype=dtype)
