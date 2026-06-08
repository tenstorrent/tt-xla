# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Text-to-image (diffusion) benchmarks.

Config-driven entry points (one ``test_<model>`` per model) that reuse the
end-to-end pipelines from ``examples/pytorch`` and drive them through the shared
two-pass harness in ``benchmarks/imagegen_benchmark.py``. This mirrors the
``test_vision.py`` / ``vision_benchmark.py`` split: model-specific config lives
here, the reusable measurement logic lives in ``benchmarks/``.

Each model runs the heavy net (UNet / MMDiT transformer) on TT and keeps the
precision-sensitive text encoders, scheduler and VAE on CPU.
"""

import json

from benchmarks.imagegen_benchmark import benchmark_imagegen_torch_xla
from utils import aggregate_ttnn_perf_metrics, resolve_display_name

# Defaults shared by all image-gen models.
DEFAULT_OPTIMIZATION_LEVEL = 1
DEFAULT_TRACE_ENABLED = False
DEFAULT_SEED = 42


def test_imagegen(
    build_pipeline_fn,
    model_info_name,
    output_file,
    prompt,
    num_inference_steps,
    height,
    width,
    request=None,
    optimization_level=DEFAULT_OPTIMIZATION_LEVEL,
    trace_enabled=DEFAULT_TRACE_ENABLED,
    output_image_path=None,
):
    """Run a text-to-image benchmark with the given configuration.

    Args:
        build_pipeline_fn: Callable returning ``(pipeline, generate_fn, save_fn)``;
            see ``benchmark_imagegen_torch_xla``.
        model_info_name: Model name for identification and reporting.
        output_file: Path to save benchmark results as JSON.
        prompt: Text prompt to generate from.
        num_inference_steps: Number of denoising steps.
        height, width: Output image dimensions.
        optimization_level: Optimization level (0, 1, or 2).
        trace_enabled: Enable trace.
        output_image_path: If set, the warm-pass image is saved here.
    """
    resolved_display_name = resolve_display_name(
        request=request, fallback=model_info_name
    )
    ttnn_perf_metrics_output_file = f"tt_xla_{resolved_display_name}_perf_metrics"

    print(f"Running image-gen benchmark for model: {model_info_name}")
    print(
        f"""Configuration:
    optimization_level={optimization_level}
    trace_enabled={trace_enabled}
    prompt={prompt!r}
    num_inference_steps={num_inference_steps}
    height={height}
    width={width}
    ttnn_perf_metrics_output_file={ttnn_perf_metrics_output_file}
    """
    )

    results = benchmark_imagegen_torch_xla(
        build_pipeline_fn=build_pipeline_fn,
        model_info_name=model_info_name,
        display_name=resolved_display_name,
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        optimization_level=optimization_level,
        trace_enabled=trace_enabled,
        ttnn_perf_metrics_output_file=ttnn_perf_metrics_output_file,
        output_image_path=output_image_path,
    )

    if output_file:
        results["project"] = "tt-forge/tt-xla"
        results["model_rawname"] = model_info_name

        aggregate_ttnn_perf_metrics(ttnn_perf_metrics_output_file, results)

        with open(output_file, "w") as file:
            json.dump(results, file, indent=2)


def test_stable_diffusion_1_5(output_file, request):
    from examples.pytorch.sd_v1_5_pipeline import SD15Config, SD15Pipeline, save_image

    prompt = "a photo of a cat"
    num_inference_steps = 50
    height = width = 512

    def build_pipeline_fn():
        # CLIP on CPU (precision-sensitive); UNet on TT. Matches the example's
        # validated configuration.
        pipeline = SD15Pipeline(config=SD15Config(device="cpu", clip_on_tt=False))
        pipeline.setup()

        def generate_fn(steps):
            return pipeline.generate(
                prompt=prompt,
                negative_prompt="",
                cfg_scale=7.5,
                num_inference_steps=steps,
                seed=DEFAULT_SEED,
            )

        return pipeline, generate_fn, save_image

    test_imagegen(
        build_pipeline_fn=build_pipeline_fn,
        model_info_name="stable-diffusion-v1-5",
        output_file=output_file,
        request=request,
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        output_image_path="test_sd15_output.png",
    )


def test_stable_diffusion_3(output_file, request):
    from examples.pytorch.sd_v3_pipeline import SD3Config, SD3Pipeline, save_image

    prompt = "An astronaut riding a green horse"
    num_inference_steps = 28
    height = width = 1024

    def build_pipeline_fn():
        pipeline = SD3Pipeline(config=SD3Config(device="cpu"))
        pipeline.setup()

        def generate_fn(steps):
            return pipeline.generate(
                prompt=prompt,
                negative_prompt="",
                guidance_scale=7.0,
                num_inference_steps=steps,
                seed=DEFAULT_SEED,
            )

        return pipeline, generate_fn, save_image

    test_imagegen(
        build_pipeline_fn=build_pipeline_fn,
        model_info_name="stable-diffusion-3-medium",
        output_file=output_file,
        request=request,
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        output_image_path="test_sd3_output.png",
    )


def test_bria_2_3(output_file, request):
    from examples.pytorch.bria_2_3_pipeline import (
        Bria23Config,
        Bria23Pipeline,
        save_image,
    )

    prompt = (
        "A portrait of a Beautiful and playful ethereal singer, "
        "golden designs, highly detailed, blurry background"
    )
    num_inference_steps = 50
    height = width = 1024

    def build_pipeline_fn():
        pipeline = Bria23Pipeline(config=Bria23Config(device="cpu"))
        pipeline.setup()

        def generate_fn(steps):
            return pipeline.generate(
                prompt=prompt,
                negative_prompt="",
                guidance_scale=5.0,
                num_inference_steps=steps,
                seed=DEFAULT_SEED,
            )

        return pipeline, generate_fn, save_image

    test_imagegen(
        build_pipeline_fn=build_pipeline_fn,
        model_info_name="bria-2.3",
        output_file=output_file,
        request=request,
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        output_image_path="test_bria_2_3_output.png",
    )
