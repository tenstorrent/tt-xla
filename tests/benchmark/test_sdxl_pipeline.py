# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json

from benchmarks.sdxl_benchmark import benchmark_sdxl_pipeline
from benchmarks.sdxl_pipeline import SDXLConstants, load_pipeline_models
from utils import aggregate_ttnn_perf_metrics, resolve_display_name


def run_sdxl_benchmark(
    output_file,
    request,
    resolution,
    num_inference_steps=SDXLConstants.NUM_INFERENCE_STEPS,
    optimization_level=SDXLConstants.OPTIMIZATION_LEVEL,
    loop_count=SDXLConstants.LOOP_COUNT,
    data_format=SDXLConstants.DATA_FORMAT,
):
    model_info_name = f"sdxl_pipeline_{resolution}x{resolution}"
    resolved_display_name = resolve_display_name(
        request=request, fallback=model_info_name
    )
    ttnn_perf_metrics_output_file = f"tt_xla_{resolved_display_name}_perf_metrics"

    if resolution == 1024:
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        hf_variant = "fp16"
    else:
        model_id = "hotshotco/SDXL-512"
        hf_variant = None

    print(f"Running SDXL pipeline benchmark: {model_info_name}")
    print(
        f"Configuration:\n"
        f"    model_id={model_id}\n"
        f"    resolution={resolution}\n"
        f"    num_inference_steps={num_inference_steps}\n"
        f"    optimization_level={optimization_level}\n"
        f"    loop_count={loop_count}\n"
        f"    data_format={data_format}\n"
        f"    ttnn_perf_metrics_output_file={ttnn_perf_metrics_output_file}"
    )

    print("Loading SDXL pipeline models...")
    unet, vae, text_encoder, text_encoder_2, tokenizer, tokenizer_2, scheduler = (
        load_pipeline_models(model_id, hf_variant)
    )

    results = benchmark_sdxl_pipeline(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        scheduler=scheduler,
        model_info_name=model_info_name,
        display_name=resolved_display_name,
        optimization_level=optimization_level,
        resolution=resolution,
        num_inference_steps=num_inference_steps,
        loop_count=loop_count,
        data_format=data_format,
        ttnn_perf_metrics_output_file=ttnn_perf_metrics_output_file,
    )

    if output_file:
        results["project"] = "tt-forge/tt-xla"
        results["model_rawname"] = model_info_name
        aggregate_ttnn_perf_metrics(ttnn_perf_metrics_output_file, results)
        with open(output_file, "w") as file:
            json.dump(results, file, indent=2)


def test_sdxl_pipeline_512(output_file, request):
    run_sdxl_benchmark(
        output_file=output_file,
        request=request,
        resolution=512,
        loop_count=1,
    )


def test_sdxl_pipeline_1024(output_file, request):
    run_sdxl_benchmark(
        output_file=output_file,
        request=request,
        resolution=1024,
        loop_count=1,
    )
