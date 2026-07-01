# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Text-to-image (diffusion) benchmarks.

Config-driven entry points (one ``test_<model>`` per model) that drive
per-model pipelines through the shared harness in
``benchmarks/imagegen_benchmark.py``. This mirrors the ``test_vision.py`` /
``vision_benchmark.py`` split: model-specific config lives here, the reusable
measurement logic lives in ``benchmarks/``.
"""

import json

import pytest
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
        build_pipeline_fn: Callable returning ``(pipeline, generate_fn)``;
            see ``benchmark_imagegen_torch_xla``.
        model_info_name: Model name for identification and reporting.
        output_file: Path to save benchmark results as JSON.
        prompt: Text prompt to generate from.
        num_inference_steps: Number of denoising steps.
        height, width: Output image dimensions.
        optimization_level: Optimization level (0, 1, or 2).
        trace_enabled: Enable trace.
        output_image_path: If set, the steady-state image is saved here.
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


def test_playground_v2_5(output_file, request):
    from benchmarks.playground_v2_5_pipeline import (
        PlaygroundV25Config,
        PlaygroundV25Pipeline,
    )

    prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
    num_inference_steps = 50
    height = width = 1024

    def build_pipeline_fn(compile_options):
        # All 4 components on TT. compile_options forwarded into Config so the
        # VAE-only opt_level switch can merge instead of clobbering.
        pipeline = PlaygroundV25Pipeline(
            config=PlaygroundV25Config(compile_options=compile_options)
        )
        pipeline.setup()

        def generate_fn(prompt, steps):
            return pipeline.generate(
                prompt=prompt,
                negative_prompt=None,
                cfg_scale=3.0,
                num_inference_steps=steps,
                seed=DEFAULT_SEED,
            )

        return pipeline, generate_fn

    test_imagegen(
        build_pipeline_fn=build_pipeline_fn,
        model_info_name="playground-v2.5",
        output_file=output_file,
        request=request,
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        # opt_level=0 for text encoders + UNet (text_encoder 1 hits
        # "Unsupported buffer type" at opt_level=1). VAE switches to
        # opt_level=1 inline (and resets after) because GroupNorm
        # decomposition at opt_level=0 OOMs the VAE.
        optimization_level=0,
        output_image_path="test_playground_v2_5_output.png",
    )


def test_sdxl_lightning(output_file, request):
    from benchmarks.sdxl_lightning_pipeline import (
        SDXLLightningConfig,
        SDXLLightningPipeline,
    )

    # SDXL-Lightning: distilled 4-step model, guidance_scale=0 (no CFG).
    prompt = "A girl smiling"
    num_inference_steps = 4
    height = width = 1024

    def build_pipeline_fn(compile_options):
        # All 4 components on TT. compile_options forwarded into Config so the
        # VAE-only opt_level switch can merge instead of clobbering.
        pipeline = SDXLLightningPipeline(
            config=SDXLLightningConfig(compile_options=compile_options)
        )
        pipeline.setup()

        def generate_fn(prompt, steps):
            return pipeline.generate(
                prompt=prompt,
                num_inference_steps=steps,
                seed=DEFAULT_SEED,
            )

        return pipeline, generate_fn

    test_imagegen(
        build_pipeline_fn=build_pipeline_fn,
        model_info_name="sdxl-lightning",
        output_file=output_file,
        request=request,
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        optimization_level=0,
        output_image_path="test_sdxl_lightning_output.png",
    )


def test_infinity_2b(output_file, request):
    from benchmarks.infinity_pipeline import InfinityConfig, InfinityPipeline

    # Infinity 2B: autoregressive next-scale prediction (not diffusion). The
    # "steps" are the fixed scale schedule (13 scales at 1M), so
    # num_inference_steps is nominal (reporting only) and ignored by generate().
    # Transformer 8-way tensor-parallel sharded on TT; T5 text encoder, sampling
    # and BSQ-VAE decode stay on CPU. The pipeline's ``_perf`` maps onto the shared
    # harness's te1/unet_steps/vae fields (te1=T5 encode, unet_steps=per-scale
    # transformer forwards, vae=BSQ-VAE decode; te2 is unused).
    prompt = "A fantasy landscape with mountains and rivers"
    num_inference_steps = 13
    height = width = 1024

    def build_pipeline_fn(compile_options):
        pipeline = InfinityPipeline(
            config=InfinityConfig(compile_options=compile_options)
        )
        pipeline.setup()

        def generate_fn(prompt, steps):
            # steps is ignored: the scale schedule is fixed by the resolution preset.
            return pipeline.generate(prompt=prompt, seed=DEFAULT_SEED)

        return pipeline, generate_fn

    test_imagegen(
        build_pipeline_fn=build_pipeline_fn,
        model_info_name="infinity-2b",
        output_file=output_file,
        request=request,
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        # opt_level=0 (matches the nightly's default and SDXL): opt_level=1
        # recompiles the sharded transformer differently and fails in CI.
        optimization_level=0,
        output_image_path="test_infinity_2b_output.png",
    )
