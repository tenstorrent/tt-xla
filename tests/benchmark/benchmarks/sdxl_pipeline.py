# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch_xla.core.xla_model as xm
from diffusers import AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer


class SDXLConstants:
    NUM_INFERENCE_STEPS = 20
    CFG_SCALE = 7.5
    PROMPT = "a photo of an astronaut riding a horse on mars"
    NEGATIVE_PROMPT = ""
    SEED = 42
    MODULE_EXPORT_PATH = "modules"
    OPTIMIZATION_LEVEL = 1
    LOOP_COUNT = 3
    DATA_FORMAT = "bfloat16"


def load_pipeline_models(model_id, variant):
    vae = AutoencoderKL.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )

    unet = UNet2DConditionModel.from_pretrained(
        model_id,
        subfolder="unet",
        variant=variant,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    text_encoder = CLIPTextModel.from_pretrained(
        model_id,
        subfolder="text_encoder",
        variant=variant,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )

    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        model_id,
        subfolder="text_encoder_2",
        variant=variant,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )

    tokenizer = CLIPTokenizer.from_pretrained(
        model_id, subfolder="tokenizer", trust_remote_code=True
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        model_id, subfolder="tokenizer_2", trust_remote_code=True
    )

    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

    return unet, vae, text_encoder, text_encoder_2, tokenizer, tokenizer_2, scheduler


def encode_prompt(
    prompt,
    negative_prompt,
    tokenizer,
    tokenizer_2,
    text_encoder,
    text_encoder_2,
    device,
):
    encoder_hidden_states = []
    pooled_text_embeds = None

    for curr_tokenizer, curr_text_encoder in [
        (tokenizer, text_encoder),
        (tokenizer_2, text_encoder_2),
    ]:
        cond_tokens = curr_tokenizer.batch_encode_plus(
            [prompt], padding="max_length", max_length=77
        ).input_ids
        uncond_tokens = curr_tokenizer.batch_encode_plus(
            [negative_prompt], padding="max_length", max_length=77
        ).input_ids

        cond_tokens = torch.tensor(cond_tokens, dtype=torch.long).to(device=device)
        uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long).to(device=device)

        cond_output = curr_text_encoder(cond_tokens, output_hidden_states=True)
        cond_hidden_state = cond_output.hidden_states[-2]

        uncond_output = curr_text_encoder(uncond_tokens, output_hidden_states=True)
        uncond_hidden_state = uncond_output.hidden_states[-2]

        if curr_text_encoder is text_encoder_2:
            pooled_cond_text_embeds = cond_output.text_embeds
            pooled_uncond_text_embeds = uncond_output.text_embeds
            pooled_text_embeds = torch.cat(
                [pooled_uncond_text_embeds, pooled_cond_text_embeds], dim=0
            )

        curr_hidden_state = torch.cat([uncond_hidden_state, cond_hidden_state], dim=0)
        encoder_hidden_states.append(curr_hidden_state)

    encoder_hidden_states = torch.cat(encoder_hidden_states, dim=-1)
    return encoder_hidden_states, pooled_text_embeds


def run_denoising_loop(
    unet,
    scheduler,
    latents,
    encoder_hidden_states,
    pooled_text_embeds,
    time_ids,
    num_inference_steps,
    cfg_scale,
):
    tt_cast = lambda x: (
        x.to(dtype=torch.bfloat16).to(device=xm.xla_device())
        if x.device == torch.device("cpu")
        else x.to(dtype=torch.bfloat16)
    )
    cpu_cast = lambda x: x.to("cpu").to(dtype=torch.float16)

    scheduler.set_timesteps(num_inference_steps)

    for i, timestep in enumerate(scheduler.timesteps):
        model_input = latents.repeat(2, 1, 1, 1)
        model_input = scheduler.scale_model_input(model_input, timestep)
        model_input = tt_cast(model_input)
        timestep_tt = tt_cast(timestep.unsqueeze(0))
        encoder_hidden_states_tt = tt_cast(encoder_hidden_states)
        pooled_text_embeds_tt = tt_cast(pooled_text_embeds)
        time_ids_tt = tt_cast(time_ids)

        unet_output = unet(
            model_input,
            timestep_tt,
            encoder_hidden_states_tt,
            added_cond_kwargs={
                "text_embeds": pooled_text_embeds_tt,
                "time_ids": time_ids_tt,
            },
        ).sample
        unet_output = cpu_cast(unet_output)

        uncond_output, cond_output = unet_output.chunk(2)
        model_output = uncond_output + (cond_output - uncond_output) * cfg_scale

        latents = cpu_cast(latents)
        latents = scheduler.step(
            model_output, cpu_cast(timestep_tt), latents
        ).prev_sample

    return latents


def run_vae_decode(vae, latents, scaling_factor):
    latents = latents / scaling_factor
    latents = latents.to(dtype=torch.float32)
    images = vae.decode(latents).sample
    return images
