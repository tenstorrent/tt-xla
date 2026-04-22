# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import socket
import time

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from benchmarks.sdxl_pipeline import (
    SDXLConstants,
    encode_prompt,
    run_denoising_loop,
    run_vae_decode,
)
from utils import (
    create_benchmark_result,
    get_benchmark_metadata,
    get_xla_device_arch,
    print_benchmark_results,
)

xr.set_device_type("TT")

_WARMUP_INFERENCE_STEPS = 5


def sdxl_inference_pass(
    unet,
    vae,
    text_encoder,
    text_encoder_2,
    tokenizer,
    tokenizer_2,
    scheduler,
    latents_h,
    latents_w,
    time_ids,
    prompt,
    negative_prompt,
    cfg_scale,
    seed,
    num_inference_steps,
    run_vae=True,
):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

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

    latents = torch.randn(
        (1, 4, latents_h, latents_w), generator=generator, dtype=torch.float16
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

    vae_start = time.perf_counter_ns()
    if run_vae:
        run_vae_decode(vae, latents, vae.config.scaling_factor)
    vae_end = time.perf_counter_ns()

    e2e_end = time.perf_counter_ns()

    return {
        "text_encode_s": (text_end - text_start) / 1e9,
        "denoise_s": (denoise_end - denoise_start) / 1e9,
        "vae_decode_s": (vae_end - vae_start) / 1e9,
        "e2e_s": (e2e_end - e2e_start) / 1e9,
    }


def benchmark_sdxl_pipeline(
    unet,
    vae,
    text_encoder,
    text_encoder_2,
    tokenizer,
    tokenizer_2,
    scheduler,
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
    latents_h = resolution // 8
    latents_w = resolution // 8

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

    target_shape = orig_shape = (resolution, resolution)
    crop_top_left = (0, 0)
    time_ids = torch.tensor(
        [*orig_shape, *crop_top_left, *target_shape], dtype=torch.float16
    ).repeat(2, 1)

    pass_kwargs = dict(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        scheduler=scheduler,
        latents_h=latents_h,
        latents_w=latents_w,
        time_ids=time_ids,
        prompt=prompt,
        negative_prompt=negative_prompt,
        cfg_scale=cfg_scale,
    )

    print("Starting warmup...")
    with torch.no_grad():
        sdxl_inference_pass(
            **pass_kwargs,
            seed=seed,
            num_inference_steps=_WARMUP_INFERENCE_STEPS,
            run_vae=False,
        )
    print("Warmup completed.")

    print(f"Starting benchmark ({loop_count} iterations)...")
    text_encode_times = []
    denoise_times = []
    vae_decode_times = []
    e2e_times = []

    with torch.no_grad():
        for i in range(loop_count):
            timings = sdxl_inference_pass(
                **pass_kwargs,
                seed=seed + i,
                num_inference_steps=num_inference_steps,
                run_vae=True,
            )
            text_encode_times.append(timings["text_encode_s"])
            denoise_times.append(timings["denoise_s"])
            vae_decode_times.append(timings["vae_decode_s"])
            e2e_times.append(timings["e2e_s"])

            print(
                f"Iteration {i}: e2e={timings['e2e_s']:.3f}s "
                f"(text={timings['text_encode_s']:.3f}s, "
                f"denoise={timings['denoise_s']:.3f}s, "
                f"vae={timings['vae_decode_s']:.3f}s)"
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
        {
            "measurement_name": "avg_e2e_latency_s",
            "value": avg_e2e,
            "step_name": model_info_name,
        },
        {
            "measurement_name": "avg_text_encode_latency_s",
            "value": avg_text_encode,
            "step_name": model_info_name,
        },
        {
            "measurement_name": "avg_denoise_latency_s",
            "value": avg_denoise,
            "step_name": model_info_name,
        },
        {
            "measurement_name": "avg_vae_decode_latency_s",
            "value": avg_vae_decode,
            "step_name": model_info_name,
        },
        {
            "measurement_name": "num_inference_steps",
            "value": num_inference_steps,
            "step_name": model_info_name,
        },
        {
            "measurement_name": "samples_per_sec",
            "value": loop_count / total_time,
            "step_name": model_info_name,
        },
        {
            "measurement_name": "per_step_denoise_latency_s",
            "value": avg_denoise / num_inference_steps,
            "step_name": model_info_name,
        },
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
