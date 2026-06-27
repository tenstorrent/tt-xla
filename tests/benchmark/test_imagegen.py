# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Text-to-image benchmarks.

Config-driven entry points (one ``test_<model>`` per model) that drive
per-model pipelines through a shared harness in ``benchmarks/``. This mirrors
the ``test_vision.py`` / ``vision_benchmark.py`` split: model-specific config
lives here, the reusable measurement logic lives in ``benchmarks/``.

Diffusion models use ``benchmarks/imagegen_benchmark.py``; autoregressive
image-token models (Janus-Pro) use ``benchmarks/ar_imagegen_benchmark.py``,
which reports decode tokens/second instead of per-step denoising metrics.
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


def _run_janus_pro_benchmark(
    model_id, model_info_name, output_image_path, output_file, request
):
    """Shared Janus-Pro AR benchmark body (1B and 7B differ only by ``model_id``).

    Unlike the diffusion models above, Janus-Pro generates the image
    autoregressively (576 image tokens), so it uses the AR harness in
    ``benchmarks/ar_imagegen_benchmark.py``. The ``janus`` runtime package is
    not in the base env, so the whole run is wrapped in ``RequirementsManager``
    (same as the nightly pipeline test).
    """
    import inspect

    from benchmarks.ar_imagegen_benchmark import benchmark_ar_imagegen_torch_xla
    from benchmarks.janus_pro_pipeline import IMG_SIZE, JanusProConfig, JanusProPipeline

    import third_party.tt_forge_models.janus_pro.text_to_image.pytorch.loader as janus_loader
    from tests.runner.requirements import RequirementsManager

    prompt = (
        "A close-up high-contrast photo of Sydney Opera House sitting next to "
        "Eiffel tower, under a blue night sky of roiling energy, exploding "
        "yellow stars, and radiating swirls of blue."
    )
    num_image_tokens = 576

    def build_pipeline_fn(compile_options):
        pipeline = JanusProPipeline(
            config=JanusProConfig(model_id=model_id, compile_options=compile_options)
        )
        pipeline.setup()

        def generate_fn(prompt, num_tokens):
            return pipeline.generate(
                prompt=prompt, num_image_tokens=num_tokens, seed=DEFAULT_SEED
            )

        return pipeline, generate_fn

    resolved_display_name = resolve_display_name(
        request=request, fallback=model_info_name
    )
    ttnn_perf_metrics_output_file = f"tt_xla_{resolved_display_name}_perf_metrics"

    loader_path = inspect.getsourcefile(janus_loader)
    with RequirementsManager.for_loader(loader_path, framework="torch"):
        results = benchmark_ar_imagegen_torch_xla(
            build_pipeline_fn=build_pipeline_fn,
            model_info_name=model_info_name,
            display_name=resolved_display_name,
            prompt=prompt,
            num_image_tokens=num_image_tokens,
            image_size=IMG_SIZE,
            # opt_level=0 matches the nightly pipeline (which runs at the
            # compiler default); opt_level=1 stalls the Janus LM/vision compile.
            optimization_level=0,
            trace_enabled=DEFAULT_TRACE_ENABLED,
            ttnn_perf_metrics_output_file=ttnn_perf_metrics_output_file,
            output_image_path=output_image_path,
        )

    if output_file:
        results["project"] = "tt-forge/tt-xla"
        results["model_rawname"] = model_info_name
        aggregate_ttnn_perf_metrics(ttnn_perf_metrics_output_file, results)
        with open(output_file, "w") as file:
            json.dump(results, file, indent=2)


def test_janus_pro(output_file, request):
    """Janus-Pro-1B autoregressive text-to-image benchmark."""
    from benchmarks.janus_pro_pipeline import REPO_ID_PRO_1B

    _run_janus_pro_benchmark(
        model_id=REPO_ID_PRO_1B,
        model_info_name="janus-pro-1b",
        output_image_path="test_janus_pro_output.png",
        output_file=output_file,
        request=request,
    )


def test_janus_pro_7b(output_file, request):
    """Janus-Pro-7B autoregressive text-to-image benchmark (blackhole).

    Skips on wormhole (n150): the 7B model OOMs the DRAM there. The matrix pins
    this entry to p150, so CI never schedules it on n150; this guard covers
    manual/general runs. Requires blackhole (p150).
    """
    import torch_xla.runtime as xr
    from benchmarks.janus_pro_pipeline import REPO_ID_PRO_7B
    from utils import get_xla_device_arch

    xr.set_device_type("TT")
    if get_xla_device_arch() == "wormhole":
        pytest.skip("Janus-Pro-7B OOMs on n150 (wormhole); requires p150 (blackhole)")

    _run_janus_pro_benchmark(
        model_id=REPO_ID_PRO_7B,
        model_info_name="janus-pro-7b",
        output_image_path="test_janus_pro_7b_output.png",
        output_file=output_file,
        request=request,
    )
