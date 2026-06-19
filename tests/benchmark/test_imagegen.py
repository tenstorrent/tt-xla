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


@pytest.mark.xfail(
    reason=(
        "VAE compile TT_FATAL during warmup (cores harvested / device_hash mismatch), "
        "possibly due to recent uplift — "
        "https://github.com/tenstorrent/tt-xla/issues/5176"
    ),
    strict=False,
)
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
        # Text encoders + UNet on TT; VAE on CPU. The VAE OOMs from GroupNorm
        # decomposition at opt_level=0; opt_level=1 enables the composite
        # ttnn.group_norm which lets the VAE pass, but switching opt level
        # (UNet opt 0 -> VAE opt 1) trips a device-hash mismatch
        # (https://github.com/tenstorrent/tt-xla/issues/5176). So keep the VAE on
        # CPU until tt-metal https://github.com/tenstorrent/tt-metal/pull/46959 lands.
        pipeline = SDXLLightningPipeline(
            config=SDXLLightningConfig(vae_on_tt=False, compile_options=compile_options)
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


def test_flux2_dev(output_file, request):
    from benchmarks.flux2_dev_pipeline import Flux2DevConfig, Flux2DevPipeline
    from third_party.tt_forge_models.flux2.pytorch.src.model_utils import (
        HEIGHT,
        NUM_INFERENCE_STEPS,
        PROMPT,
        WIDTH,
    )

    # Composite FLUX.2-dev: denoiser tensor-parallel on device (full mesh),
    # text encoder (~24B) and VAE on CPU. Bringup resolution is 128x128 with a
    # short schedule (see model_utils); override via FLUX2_* env vars.
    prompt = PROMPT
    num_inference_steps = NUM_INFERENCE_STEPS
    height = HEIGHT
    width = WIDTH

    def build_pipeline_fn(compile_options):
        pipeline = Flux2DevPipeline(
            config=Flux2DevConfig(compile_options=compile_options)
        )
        pipeline.setup()

        def generate_fn(prompt, steps):
            return pipeline.generate(prompt=prompt, num_inference_steps=steps)

        return pipeline, generate_fn

    test_imagegen(
        build_pipeline_fn=build_pipeline_fn,
        model_info_name="flux2-dev",
        output_file=output_file,
        request=request,
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        # Winning config from model-perf-tuning: opt=0, trace=False (baseline).
        # opt>=1 aborts with an OpModel worker-grid mismatch (system desc 10x11 vs
        # device 10x13); trace and bfp_bf8 weights were both perf-neutral at the
        # 128x128 bringup resolution. See the tuning report / branch git log.
        optimization_level=0,
        trace_enabled=False,
        output_image_path="test_flux2_dev_output.png",
    )
