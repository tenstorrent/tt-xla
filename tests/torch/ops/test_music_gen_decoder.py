# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import Framework, run_op_test_with_random_inputs
from utils import Category
from third_party.tt_forge_models.stereo.pytorch import ModelLoader,ModelVariant
from tests.infra.testers.compiler_config import CompilerConfig
from tests.infra.testers.single_chip.op.op_tester import run_op_test_with_saved_inputs
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache
import torch.nn as nn
import random
class DecoderWithEarlyStop(torch.nn.Module):
    def __init__(self, decoder, stop_at_layer):
        super().__init__()
        self.decoder = decoder
        self.stop_at_layer = stop_at_layer

        # Explicitly forward attributes needed in forward
        self.num_codebooks = decoder.num_codebooks
        self.embed_tokens = decoder.embed_tokens
        self.layer_norm = decoder.layer_norm
        self.layers = decoder.layers
        self.embed_positions = decoder.embed_positions
        self.dropout = decoder.dropout
        self.gradient_checkpointing = getattr(decoder, "gradient_checkpointing", False)
        self.config = decoder.config
        self.training=decoder.training
        self.layerdrop=decoder.layerdrop


    def forward(self,input_ids,attention_mask,encoder_attention_mask,encoder_hidden_states,head_mask,cross_attn_head_mask,past_key_values,inputs_embeds,use_cache,output_attentions,output_hidden_states,return_dict,cache_position):
        """
        Wrapper around HF MusicgenDecoder forward to add early stop after a specific layer.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            # (bsz * codebooks, seq_len) -> (bsz, codebooks, seq_len)
            input = input_ids.reshape(-1, self.num_codebooks, input_ids.shape[-1])
            bsz, num_codebooks, seq_len = input.shape
            input_shape = (bsz, seq_len)
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            input = inputs_embeds[:, :, -1:]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing`. Setting `use_cache=False`..."
                )
                use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = EncoderDecoderCache(DynamicCache(config=self.config), DynamicCache(config=self.config))
        if use_cache and isinstance(past_key_values, tuple):
            logger.warning_once(
                "Passing a tuple of `past_key_values` is deprecated and will be removed in Transformers v4.58.0. "
                "You should pass an instance of `EncoderDecoderCache` instead, e.g. "
                "`past_key_values=EncoderDecoderCache.from_legacy_cache(past_key_values)`."
            )
            past_key_values = EncoderDecoderCache.from_legacy_cache(past_key_values)

        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = sum([self.embed_tokens[codebook](input[:, codebook]) for codebook in range(num_codebooks)])

        attention_mask = self.decoder._update_causal_mask(
            attention_mask,
            input_shape,
            inputs_embeds,
            past_key_values_length,
        )
        encoder_attention_mask = self.decoder._update_cross_attn_mask(
            encoder_hidden_states,
            encoder_attention_mask,
            input_shape,
            inputs_embeds,
        )

        # embed positions
        positions = self.embed_positions(input, past_key_values_length)
        hidden_states = inputs_embeds + positions.to(inputs_embeds.device)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != len(self.layers):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {attn_mask.size()[0]}."
                    )
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://huggingface.co/papers/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask,
                encoder_hidden_states,  # as a positional argument for gradient checkpointing
                encoder_attention_mask=encoder_attention_mask,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                cross_attn_layer_head_mask=(cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None),
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                # layer_num = idx
            )
            
            hidden_states = layer_outputs[0]
            # NOTE (test-only):
            # The following early-return logic is NOT part of the original HuggingFace
            # MusicgenDecoder implementation. It is added only for unit testing / PCC
            # validation purposes, to stop the decoder execution after a specific
            # decoder layer (e.g., layer 2 for LARGE, layer 3 for MEDIUM) and avoid
            # running subsequent layers or producing their past_key_values.
            #
            # This allows isolating and validating intermediate decoder outputs without
            # modifying the upstream HuggingFace source or affecting full-model behavior.
            if idx==self.stop_at_layer:
                # Test-only early exit:
                # Not part of the original HF MusicgenDecoder. Used to stop decoder execution
                # after a specific layer to isolate intermediate outputs and PKVs for PCC checks.       
                return hidden_states,past_key_values

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        hidden_states = self.layer_norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )



class decoderrr(torch.nn.Module):
    def __init__(self,model,stop_at_layer):
        super().__init__()
        self.decoderr = DecoderWithEarlyStop(model,stop_at_layer)

    def forward(self, input_ids,attention_mask,encoder_attention_mask,encoder_hidden_states,head_mask,cross_attn_head_mask,past_key_values,inputs_embeds,use_cache,output_attentions,output_hidden_states,return_dict,cache_position):
        return self.decoderr(input_ids=input_ids,attention_mask=attention_mask,encoder_attention_mask=encoder_attention_mask,encoder_hidden_states=encoder_hidden_states,head_mask=head_mask,cross_attn_head_mask=cross_attn_head_mask,past_key_values=past_key_values,inputs_embeds=inputs_embeds,use_cache=use_cache,output_attentions=output_attentions,output_hidden_states=output_hidden_states,return_dict=return_dict,cache_position=cache_position)

@pytest.mark.parametrize(
    "variant,layer", # layer where PCC Drop is observed for Past Key Values during whole model run for specific variant
    [
        (ModelVariant.MEDIUM, 3),
        (ModelVariant.LARGE, 2),
    ],
)
def test_decoder(variant,layer):
    loader = ModelLoader(variant)
    model = loader.load_model()
    decoder_module = model.decoder.model.decoder
    data = torch.load(f"{variant}_variant_model_decoder_input_data.pt", map_location="cpu",weights_only=False)
    run_op_test_with_saved_inputs(
        decoderrr(decoder_module,layer),
        [
            data["input_ids"],
            data["attention_mask"],
            data["encoder_attention_mask"],
            data["encoder_hidden_states"],
            data["head_mask"],
            data["cross_attn_head_mask"],
            data["past_key_values"],
            data["inputs_embeds"],
            data["use_cache"],
            data["output_attentions"],
            data["output_hidden_states"],
            data["return_dict"],
            data["cache_position"]
        ],
        framework=Framework.TORCH,
    )

