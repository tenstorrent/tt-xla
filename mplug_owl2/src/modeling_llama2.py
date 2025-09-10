# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
LLaMA modifications adding multimodal and modality-adaptive attention for mPLUG-Owl2.

# code apapted from :
# https://github.com/X-PLUG/mPLUG-Owl/tree/main/mPLUG-Owl2

MIT License

Copyright (c) 2022 mPLUG

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""


import inspect
from functools import partial
from typing import Optional, Union

import torch
import torch.utils.checkpoint
from torch import nn

import transformers
from transformers.models.llama.modeling_llama import *

from .configuration_mplug_owl2 import LlamaConfig
from .multiway import MultiwayNetwork
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.utils.deprecation import deprecate_kwarg
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.cache_utils import Cache
from typing import Callable, Optional, Union
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.masking_utils import create_causal_mask
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    LlamaRMSNorm,
    eager_attention_forward,
    apply_rotary_pos_emb,
    LlamaMLP,
)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        # Modified: Standard q_proj, MultiwayNetwork for k_proj and v_proj
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )

        # Key change: Use MultiwayNetwork for k_proj and v_proj
        self.k_proj = MultiwayNetwork(
            module_provider=partial(
                nn.Linear,
                in_features=config.hidden_size,
                out_features=config.num_key_value_heads * self.head_dim,
                bias=config.attention_bias,
            )
        )
        self.v_proj = MultiwayNetwork(
            module_provider=partial(
                nn.Linear,
                in_features=config.hidden_size,
                out_features=config.num_key_value_heads * self.head_dim,
                bias=config.attention_bias,
            )
        )

        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        modality_indicators: Optional[
            torch.Tensor
        ] = None,  # Added from modified version
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Modified: Pass modality_indicators to k_proj and v_proj
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # Key changes: Use modality_indicators for k_proj and v_proj
        if modality_indicators is not None:
            key_states = (
                self.k_proj(hidden_states, modality_indicators)
                .view(hidden_shape)
                .transpose(1, 2)
            )
            value_states = (
                self.v_proj(hidden_states, modality_indicators)
                .view(hidden_shape)
                .transpose(1, 2)
            )
        else:
            # Fallback for backward compatibility
            key_states = (
                self.k_proj(hidden_states, None).view(hidden_shape).transpose(1, 2)
            )
            value_states = (
                self.v_proj(hidden_states, None).view(hidden_shape).transpose(1, 2)
            )

        # Use 4.56.0's position_embeddings approach
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        # Use 4.56.0's Cache system
        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # Use 4.56.0's attention interface system
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class LlamaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        # Modified attention with modality support
        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        # Dynamic MLP initialization (from modified version)
        mlp_kwargs = {
            "config": config,
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "hidden_act": config.hidden_act,
        }
        valid_params = set(inspect.signature(LlamaMLP.__init__).parameters.keys()) - {
            "self"
        }
        mlp_kwargs = {k: v for k, v in mlp_kwargs.items() if k in valid_params}
        self.mlp = LlamaMLP(**mlp_kwargs)

        # MultiwayNetwork for layer norms (key change from modified version)
        self.input_layernorm = MultiwayNetwork(
            module_provider=partial(
                LlamaRMSNorm, hidden_size=config.hidden_size, eps=config.rms_norm_eps
            )
        )
        self.post_attention_layernorm = MultiwayNetwork(
            module_provider=partial(
                LlamaRMSNorm, hidden_size=config.hidden_size, eps=config.rms_norm_eps
            )
        )

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        modality_indicators: Optional[
            torch.Tensor
        ] = None,  # Added from modified version
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states

        # Modified to pass modality_indicators
        hidden_states = self.input_layernorm(hidden_states, modality_indicators)

        # Self Attention - modified to pass modality_indicators
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            modality_indicators=modality_indicators,  # Added
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        # Modified to pass modality_indicators
        hidden_states = self.post_attention_layernorm(
            hidden_states, modality_indicators
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


def model_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    modality_indicators: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> BaseModelOutputWithPast:
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.config)

    if cache_position is None:
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        cache_position: torch.Tensor = torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    # Use modern create_causal_mask instead of _prepare_decoder_attention_mask
    causal_mask = create_causal_mask(
        config=self.config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    hidden_states = inputs_embeds

    # Create rotary embedding if not exists and compute position embeddings
    if not hasattr(self, "rotary_emb"):
        self.rotary_emb = LlamaRotaryEmbedding(self.config)
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    if self.gradient_checkpointing and self.training:
        if use_cache:
            # logger.warning_once(
            #     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            # )
            use_cache = False

    # Modern decoder layers approach
    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        hidden_states = decoder_layer(
            hidden_states,
            modality_indicators=modality_indicators,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    hidden_states = self.norm(hidden_states)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
    )


def causal_model_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    modality_indicators: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    **kwargs: Unpack[TransformersKwargs],
) -> CausalLMOutputWithPast:
    r"""
    Forward pass for causal language modeling with multimodal support.

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of input sequence tokens in the vocabulary.
        modality_indicators (`torch.Tensor`, *optional*):
            Tensor indicating modality for each token (specific to mPlug-Owl2 multimodal processing).
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence token in the position embeddings.
        past_key_values (`Cache`, *optional*):
            Pre-computed hidden-states (key and value states) that can be used to speed up sequential decoding.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100. Tokens with indices set to `-100` are ignored (masked).
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence.
        logits_to_keep (`int` or `torch.Tensor`, *optional*, defaults to 0):
            Number of logits to keep from the end of the sequence for efficiency. If 0, keeps all logits.

    Returns:
        `CausalLMOutputWithPast`: A structured output containing loss, logits, past_key_values, hidden_states, and attentions.

    Example:

    ```python
    >>> from transformers import AutoTokenizer, LlamaForCausalLM

    >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
    >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""

    # Modern approach: direct model call with structured output
    outputs: BaseModelOutputWithPast = self.model(
        input_ids=input_ids,
        modality_indicators=modality_indicators,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state

    # Modern selective logits computation - only compute necessary logits
    slice_indices = (
        slice(-logits_to_keep, None)
        if isinstance(logits_to_keep, int)
        else logits_to_keep
    )
    logits = self.lm_head(hidden_states[:, slice_indices, :])

    loss = None
    if labels is not None:
        # Modern loss computation
        loss = self.loss_function(
            logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs
        )

    # Modern approach: always return structured output
    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def replace_llama_modality_adaptive():
    transformers.models.llama.configuration_llama.LlamaConfig = LlamaConfig
    transformers.models.llama.modeling_llama.LlamaAttention = LlamaAttention
    transformers.models.llama.modeling_llama.LlamaDecoderLayer = LlamaDecoderLayer
    transformers.models.llama.modeling_llama.LlamaModel.forward = model_forward
    transformers.models.llama.modeling_llama.LlamaForCausalLM.forward = (
        causal_model_forward
    )


if __name__ == "__main__":
    replace_llama_modality_adaptive()
    config = transformers.LlamaConfig.from_pretrained(
        "/cpfs01/shared/public/test/vicuna-7b-v1.5/"
    )
    model = transformers.LlamaForCausalLM(config)
    print(model)
