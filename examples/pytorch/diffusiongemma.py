# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DiffusionGemma 26B -- block-diffusion text generation demo on Tenstorrent hardware.

The block-diffusion driver (sampler, stopping, cache, RNG) runs on the HOST; only the two
NN components run on TT: the encoder prefill (model.model.encoder) and the decoder forward
(model.forward, input_ids=None -> decoder + lm_head). The decoded text is the model's real
on-device answer. The 26B model runs sharded (SPMD) via the loader's get_mesh_config /
load_shard_spec (MLP col->row + MoE experts; attention replicated).

Run: python examples/pytorch/diffusiongemma.py
"""

import math
import os

import numpy as np
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from loguru import logger
from torch_xla.distributed.spmd import Mesh
from tt_torch.moe_backend import TT_MOE_BACKEND_NAME  # importing registers the tt_moe backend

from third_party.tt_forge_models.diffusiongemma.pytorch import ModelLoader

PROMPT = "Why is the sky blue?"
MAX_NEW_TOKENS = 256  # one canvas block; bump to 512 for two blocks
SEED = 0


def enable_spmd():
    # tt-mlir's stablehlo pipeline expects Shardy annotations from pytorch/xla.
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()


def make_mesh(mesh_shape, mesh_names) -> Mesh:
    device_ids = np.array(range(xr.global_runtime_device_count()))
    return Mesh(device_ids, mesh_shape, mesh_names)


def to_device(obj, device):
    """Recursively move tensors in a tensor / dict / list / tuple to ``device``."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device(v, device) for v in obj)
    return obj


@torch.no_grad()
def manual_generate(
    model,
    input_ids,
    attention_mask,
    max_new_tokens,
    *,
    encoder_forward=None,
    decoder_forward=None,
    **model_kwargs,
):
    """With-cache mimic of generate() (verified bit-identical). Only the encoder prefill and
    decoder forward are swappable (default to the model's own) for TT injection."""
    encoder_forward = encoder_forward or model.model.encoder
    decoder_forward = decoder_forward or model.forward

    gen_cfg, model_kwargs = model._prepare_generation_config(
        None, max_new_tokens=max_new_tokens, **model_kwargs
    )
    batch_size, cur_len = input_ids.shape
    max_length, max_new_tokens = model._prepare_generated_length(gen_cfg, cur_len)
    max_new_canvases = math.ceil(max_new_tokens / model.config.canvas_length)

    device = input_ids.device
    canvas_length = model.config.canvas_length
    finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=device)
    past_key_values = model._prepare_cache_for_generation(
        generation_config=gen_cfg, batch_size=batch_size, max_length=max_length - canvas_length
    )
    eos_tensor = (
        torch.tensor(gen_cfg.eos_token_id, device=device)
        if gen_cfg.eos_token_id is not None
        else None
    )
    encoder_position_ids = torch.arange(
        cur_len - input_ids.shape[1], cur_len, dtype=torch.int32, device=device
    ).unsqueeze(0)
    decoder_position_ids = torch.arange(
        cur_len, cur_len + canvas_length, dtype=torch.int32, device=device
    ).unsqueeze(0)
    decoder_attention_mask = torch.nn.functional.pad(
        attention_mask, (0, canvas_length), value=True
    )

    sampler = model._prepare_sampler(gen_cfg)
    logits_processor = model._prepare_logits_processor(gen_cfg, None)
    ar_stopping = model._prepare_ar_stopping_criteria(gen_cfg, None)
    diffusion_stopping = model._prepare_diffusion_stopping_criteria(gen_cfg)

    is_prefill = True
    for block in range(max_new_canvases):
        unprocessed_input_ids, encoder_mask_mapping = model._prepare_encoder_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_position_ids=encoder_position_ids,
            past_key_values=past_key_values,
            is_prefill=is_prefill,
            canvas_length=canvas_length,
            batch_size=batch_size,
            **model_kwargs,
        )
        encoder_outputs = encoder_forward(
            input_ids=unprocessed_input_ids,
            attention_mask=encoder_mask_mapping,
            past_key_values=past_key_values,
            position_ids=encoder_position_ids,
            **model_kwargs,
        )
        past_key_values = encoder_outputs.past_key_values
        is_prefill = False

        current_canvas, self_conditioning_logits, mask_mapping, finished_denoising = (
            model._prepare_denoiser_inputs(
                decoder_attention_mask=decoder_attention_mask,
                past_key_values=past_key_values,
                sampler=sampler,
                diffusion_stopping_criteria=diffusion_stopping,
                batch_size=batch_size,
                device=device,
                model_kwargs=model_kwargs,
            )
        )
        argmax_canvas = current_canvas

        for cur_step in reversed(range(1, gen_cfg.max_denoising_steps + 1)):
            current_canvas, argmax_canvas, self_conditioning_logits, finished_denoising = (
                model._denoising_step(
                    decoder_forward=decoder_forward,
                    current_canvas=current_canvas,
                    argmax_canvas=argmax_canvas,
                    input_ids=input_ids,
                    decoder_position_ids=decoder_position_ids,
                    self_conditioning_logits=self_conditioning_logits,
                    mask_mapping=mask_mapping,
                    past_key_values=past_key_values,
                    finished_denoising=finished_denoising,
                    cur_step=cur_step,
                    sampler=sampler,
                    logits_processor=logits_processor,
                    diffusion_stopping_criteria=diffusion_stopping,
                    **model_kwargs,
                )
            )
            if torch.all(finished_denoising):
                break

        logger.info("block {}/{} done", block + 1, max_new_canvases)
        input_ids = torch.cat([input_ids, argmax_canvas], dim=-1)
        input_ids, finished_sequences = model._finalize_canvas(
            input_ids=input_ids,
            finished_sequences=finished_sequences,
            generation_config=gen_cfg,
            stopping_criteria=ar_stopping,
            canvas_length=canvas_length,
            eos_tensor=eos_tensor,
        )
        if torch.all(finished_sequences):
            break
        (
            cur_len,
            decoder_attention_mask,
            attention_mask,
            encoder_position_ids,
            decoder_position_ids,
        ) = model._prepare_kwargs_for_next_canvas(
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            past_key_values=past_key_values,
            canvas_length=canvas_length,
            cur_len=cur_len,
            is_compiling=False,
        )

    return input_ids


def run_diffusiongemma(prompt: str = PROMPT, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """Generate text with only the NN components on TT; return the decoded output."""
    enable_spmd()

    loader = ModelLoader()
    model = loader.load_model(dtype_override=torch.bfloat16)
    model.eval()
    model.config._experts_implementation = TT_MOE_BACKEND_NAME  # matches runner's inject_custom_moe
    inputs = loader.load_inputs(dtype_override=torch.bfloat16, prompt=prompt)
    # generate()'s extra inputs (e.g. mm_token_type_ids), minus decoder_input_ids (loop inits its canvas).
    extra_kwargs = {
        k: v
        for k, v in inputs.items()
        if k not in ("input_ids", "attention_mask", "decoder_input_ids")
    }

    # Shard weights (encoder + decoder text layers) across the mesh, then compile.
    model = model.to(xm.xla_device())
    mesh = make_mesh(*loader.get_mesh_config(xr.global_runtime_device_count()))
    xs.set_global_mesh(mesh)  # tt_moe reads get_global_mesh() for the expert-parallel axis
    for tensor, spec in loader.load_shard_spec(model).items():
        xs.mark_sharding(tensor, mesh, spec)

    # encoder_tt: encoder prefill. model_tt: decoder path (called with input_ids=None, which
    # prunes the encoder branch -> decoder + lm_head only; encoder compiled once, not twice).
    encoder_tt = torch.compile(model.model.encoder, backend="tt")
    model_tt = torch.compile(model, backend="tt")

    xla = xm.xla_device()

    def encoder_forward(**kw):
        return encoder_tt(**to_device(kw, xla))

    def decoder_forward(**kw):
        out = model_tt(input_ids=None, **to_device(kw, xla))
        out.logits = out.logits.to("cpu")
        return out

    torch.manual_seed(SEED)
    output = manual_generate(
        model,
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        encoder_forward=encoder_forward,
        decoder_forward=decoder_forward,
        **extra_kwargs,
    )
    return loader.processor.decode(output[0], skip_special_tokens=True)


if __name__ == "__main__":
    xr.set_device_type("TT")
    text = run_diffusiongemma()
    logger.info("DiffusionGemma output:\n{}", text)
