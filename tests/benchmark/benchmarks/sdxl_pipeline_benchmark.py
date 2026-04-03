# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import socket
import time

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from diffusers import AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from utils import (
    create_benchmark_result,
    get_benchmark_metadata,
    get_xla_device_arch,
    print_benchmark_results,
)

xr.set_device_type("TT")


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
    PERF_REGRESSION_MARGIN_PERCENT = 0.1 # margin for hardware variability
    CHECKED_METRICS = frozenset({"avg_denoise_latency_s", "per_step_denoise_latency_s"})
    PERF_THRESHOLDS = {
        
        ("p100", 512): {
            "avg_denoise_latency_s": 5.722*(1+PERF_REGRESSION_MARGIN_PERCENT),
            "per_step_denoise_latency_s": 0.2861 *(1+PERF_REGRESSION_MARGIN_PERCENT)
        },
        ("p150", 1024): {
            "avg_denoise_latency_s": 19.522*(1+PERF_REGRESSION_MARGIN_PERCENT),
            "per_step_denoise_latency_s": 0.9761*(1+PERF_REGRESSION_MARGIN_PERCENT)
        },
    }


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


def benchmark_sdxl_pipeline(
    model_info_name,
    display_name,
    optimization_level,
    resolution,
    num_inference_steps,
    loop_count,
    data_format,
    ttnn_perf_metrics_output_file,
    cfg_scale=SDXLConstants.CFG_SCALE,
    prompt=SDXLConstants.PROMPT,
    negative_prompt=SDXLConstants.NEGATIVE_PROMPT,
    seed=SDXLConstants.SEED,
):
    if resolution == 1024:
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        hf_variant = "fp16"
    else:
        model_id = "hotshotco/SDXL-512"
        hf_variant = None

    latents_h = resolution // 8
    latents_w = resolution // 8

    print("Loading SDXL pipeline models...")
    unet, vae, text_encoder, text_encoder_2, tokenizer, tokenizer_2, scheduler = (
        load_pipeline_models(model_id, hf_variant)
    )

    options = {
        "optimization_level": optimization_level,
        "export_path": SDXLConstants.MODULE_EXPORT_PATH,
        "export_model_name": display_name or model_info_name,
        "ttnn_perf_metrics_enabled": True,
        "ttnn_perf_metrics_output_file": ttnn_perf_metrics_output_file,
    }
    torch_xla.set_custom_compile_options(options)

    unet.compile(backend="tt")
    unet = unet.to(xm.xla_device())

    print("Starting warmup...")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    with torch.no_grad():
        warmup_encoder_hidden_states, warmup_pooled = encode_prompt(
            prompt,
            negative_prompt,
            tokenizer,
            tokenizer_2,
            text_encoder,
            text_encoder_2,
            "cpu",
        )
        warmup_latents = torch.randn(
            (1, 4, latents_h, latents_w), generator=generator, dtype=torch.float16
        )
        warmup_latents = warmup_latents * scheduler.init_noise_sigma
        target_shape = orig_shape = (resolution, resolution)
        crop_top_left = (0, 0)
        time_ids = torch.tensor(
            [*orig_shape, *crop_top_left, *target_shape], dtype=torch.float16
        ).repeat(2, 1)

        run_denoising_loop(
            unet,
            scheduler,
            warmup_latents,
            warmup_encoder_hidden_states,
            warmup_pooled,
            time_ids,
            min(5, num_inference_steps),
            cfg_scale,
        )
    print("Warmup completed.")

    print(f"Starting benchmark ({loop_count} iterations)...")
    text_encode_times = []
    denoise_times = []
    vae_decode_times = []
    e2e_times = []

    with torch.no_grad():
        for i in range(loop_count):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed + i)

            e2e_start = time.perf_counter_ns()

            text_start = time.perf_counter_ns()
            encoder_hidden_states, pooled_text_embeds = encode_prompt(
                prompt,
                negative_prompt,
                tokenizer,
                tokenizer_2,
                text_encoder,
                text_encoder_2,
                "cpu",
            )
            text_end = time.perf_counter_ns()
            text_encode_times.append((text_end - text_start) / 1e9)

            latents = torch.randn(
                (1, 4, latents_h, latents_w),
                generator=generator,
                dtype=torch.float16,
            )
            latents = latents * scheduler.init_noise_sigma

            denoise_start = time.perf_counter_ns()
            latents = run_denoising_loop(
                unet,
                scheduler,
                latents,
                encoder_hidden_states,
                pooled_text_embeds,
                time_ids,
                num_inference_steps,
                cfg_scale,
            )
            denoise_end = time.perf_counter_ns()
            denoise_times.append((denoise_end - denoise_start) / 1e9)

            vae_start = time.perf_counter_ns()
            images = run_vae_decode(vae, latents, vae.config.scaling_factor)
            vae_end = time.perf_counter_ns()
            vae_decode_times.append((vae_end - vae_start) / 1e9)

            e2e_end = time.perf_counter_ns()
            e2e_times.append((e2e_end - e2e_start) / 1e9)

            print(
                f"Iteration {i}: e2e={e2e_times[-1]:.3f}s "
                f"(text={text_encode_times[-1]:.3f}s, "
                f"denoise={denoise_times[-1]:.3f}s, "
                f"vae={vae_decode_times[-1]:.3f}s)"
            )

    print("Benchmark completed.")

    avg_text_encode = sum(text_encode_times) / loop_count
    avg_denoise = sum(denoise_times) / loop_count
    avg_vae_decode = sum(vae_decode_times) / loop_count
    avg_e2e = sum(e2e_times) / loop_count
    total_time = sum(e2e_times)

    metadata = get_benchmark_metadata()

    model_type = "Image Generation, SDXL Pipeline"
    dataset_name = "Text Prompt"
    input_size = (3, resolution, resolution)

    print_benchmark_results(
        model_title=model_info_name,
        full_model_name=model_info_name,
        model_type=model_type,
        dataset_name=dataset_name,
        date=metadata["date"],
        machine_name=metadata["machine_name"],
        total_time=total_time,
        total_samples=loop_count,
        samples_per_sec=loop_count / total_time,
        batch_size=1,
        data_format=data_format,
        input_size=input_size,
    )

    custom_measurements = [
        {"measurement_name": "avg_e2e_latency_s", "value": avg_e2e, "step_name": model_info_name},
        {"measurement_name": "avg_text_encode_latency_s", "value": avg_text_encode, "step_name": model_info_name},
        {"measurement_name": "avg_denoise_latency_s", "value": avg_denoise, "step_name": model_info_name},
        {"measurement_name": "avg_vae_decode_latency_s", "value": avg_vae_decode, "step_name": model_info_name},
        {"measurement_name": "num_inference_steps", "value": num_inference_steps, "step_name": model_info_name},
        {"measurement_name": "samples_per_sec", "value": loop_count / total_time, "step_name": model_info_name},
        {"measurement_name": "per_step_denoise_latency_s", "value": avg_denoise / num_inference_steps, "step_name": model_info_name},
    ]

    result = create_benchmark_result(
        full_model_name=model_info_name,
        model_type=model_type,
        dataset_name=dataset_name,
        num_layers=-1,
        batch_size=1,
        input_size=input_size,
        loop_count=loop_count,
        data_format=data_format,
        total_time=total_time,
        total_samples=loop_count,
        custom_measurements=custom_measurements,
        optimization_level=optimization_level,
        trace_enabled=False,
        model_info=model_info_name,
        display_name=display_name,
        torch_xla_enabled=True,
        backend="tt",
        device_name=socket.gethostname(),
        arch=get_xla_device_arch(),
        device_count=xr.global_runtime_device_count(),
    )

    return result
