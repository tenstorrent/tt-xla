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

from third_party.tt_forge_models.stable_diffusion_1_5.pytorch.pipeline import (
    SD15Config,
    SD15Pipeline,
)
from third_party.tt_forge_models.stable_diffusion_3.pytorch.pipeline import (
    SD3Config,
    SD3Pipeline,
)

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


def test_stable_diffusion_1_5(output_file, request):
    prompt = "a photo of a cat"
    num_inference_steps = 50
    height = width = 512

    def build_pipeline_fn(compile_options):
        # Heavy net (UNet) on TT; precision-sensitive CLIP, scheduler and VAE on
        # CPU. compile_options is already applied globally by the harness.
        pipeline = SD15Pipeline(config=SD15Config(clip_on_tt=False))
        pipeline.setup()

        def generate_fn(prompt, steps):
            return pipeline.generate(
                prompt=prompt,
                negative_prompt="",
                cfg_scale=7.5,
                num_inference_steps=steps,
                seed=DEFAULT_SEED,
            )

        return pipeline, generate_fn

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
    prompt = "An astronaut riding a green horse"
    num_inference_steps = 28
    height = width = 1024

    def build_pipeline_fn(compile_options):
        # Heavy net (MMDiT transformer) on TT; the three text encoders,
        # scheduler and VAE on CPU.
        pipeline = SD3Pipeline(config=SD3Config())
        pipeline.setup()

        def generate_fn(prompt, steps):
            return pipeline.generate(
                prompt=prompt,
                negative_prompt="",
                guidance_scale=7.0,
                num_inference_steps=steps,
                seed=DEFAULT_SEED,
            )

        return pipeline, generate_fn

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


def test_flux2(output_file, request):
    from benchmarks.flux2_pipeline import Flux2Config, Flux2Pipeline_TT

    from third_party.tt_forge_models.flux2.pytorch.src.model_utils import (
        HEIGHT,
        PROMPT,
        WIDTH,
    )

    # FLUX.2-dev: ~24B Mistral3 text encoder + ~32B Flux2 transformer (both
    # tensor-parallel sharded across the mesh's model axis) + replicated VAE.
    # Multichip — wired to the 4-chip blackhole (qb2) in perf-bench-matrix.json.
    prompt = PROMPT
    num_inference_steps = 50
    height = HEIGHT
    width = WIDTH

    def build_pipeline_fn(compile_options):
        pipeline = Flux2Pipeline_TT(config=Flux2Config(compile_options=compile_options))
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
        model_info_name="flux2",
        output_file=output_file,
        request=request,
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        optimization_level=0,
        output_image_path="test_flux2_output.png",
    )


def test_flux(output_file, request):
    from benchmarks.flux_pipeline import FluxConfig, FluxPipeline_TT

    from third_party.tt_forge_models.flux.pytorch.src.model_utils import (
        HEIGHT,
        PROMPT,
        WIDTH,
    )

    # FLUX.1-dev native: 1024x1024, 50 steps, guidance 3.5, seq-512 (source
    # inference pipeline defaults). All 4 components (CLIP, T5, transformer, VAE)
    # run on TT; the transformer is tensor-parallel sharded across the mesh's
    # model axis. Multichip — wired to the 4-chip blackhole in
    # perf-bench-matrix.json.
    prompt = PROMPT
    num_inference_steps = 50
    height = HEIGHT
    width = WIDTH

    def build_pipeline_fn(compile_options):
        pipeline = FluxPipeline_TT(config=FluxConfig(compile_options=compile_options))
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
        model_info_name="flux1-dev",
        output_file=output_file,
        request=request,
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        optimization_level=0,
        output_image_path="test_flux1_output.png",
    )


def test_zimage(output_file, request):
    from benchmarks.zimage_pipeline import ZImageConfig, ZImagePipeline_TT

    from third_party.tt_forge_models.z_image.pytorch.src.model_utils import (
        GUIDANCE_SCALE,
        HEIGHT,
        NUM_INFERENCE_STEPS,
        PROMPT,
        WIDTH,
    )

    # Z-Image: ~6.2B ZImageTransformer2DModel + Qwen3 text encoder + VAE, all on
    # one Blackhole chip (OOMs on single Wormhole). CFG runs as two batch=1
    # passes. Blackhole-only — wired to p150-perf in perf-bench-matrix.json.
    prompt = PROMPT
    num_inference_steps = NUM_INFERENCE_STEPS
    height = HEIGHT
    width = WIDTH

    def build_pipeline_fn(compile_options):
        pipeline = ZImagePipeline_TT(
            config=ZImageConfig(compile_options=compile_options)
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
        model_info_name="zimage",
        output_file=output_file,
        request=request,
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        # opt_level=1 keeps GroupNorm as native ttnn.group_norm so the VAE decode
        # at 1280x720 does not OOM (issue #4755).
        optimization_level=1,
        output_image_path="test_zimage_output.png",
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


def test_infinity_2b(output_file, request):
    from benchmarks.infinity_pipeline import InfinityConfig, InfinityPipeline

    # Infinity 2B: autoregressive next-scale prediction (not diffusion). The
    # "steps" are the fixed scale schedule (13 scales at 1M), so
    # num_inference_steps is nominal (reporting only) and ignored by generate().
    # Transformer tensor-parallel sharded on TT; T5 text encoder, sampling
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
