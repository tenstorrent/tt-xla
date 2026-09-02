# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Text-to-video benchmarks.

One ``test_<model>`` per model; model-specific config lives here, the reusable
measurement logic in ``benchmarks/video_gen_pipeline_benchmark.py``.
"""

import json

from benchmarks.video_gen_pipeline_benchmark import (
    benchmark_video_gen_pipeline_torch_xla,
)
from utils import aggregate_ttnn_perf_metrics, resolve_display_name

DEFAULT_OPTIMIZATION_LEVEL = 0
DEFAULT_TRACE_ENABLED = False
DEFAULT_SEED = 42
DEFAULT_WARMUP_STEPS = 1


def _run_video_gen(
    build_pipeline_fn,
    model_info_name,
    output_file,
    prompt,
    num_inference_steps,
    height,
    width,
    num_frames,
    fps,
    request=None,
    optimization_level=DEFAULT_OPTIMIZATION_LEVEL,
    trace_enabled=DEFAULT_TRACE_ENABLED,
    output_video_path=None,
    warmup_steps=DEFAULT_WARMUP_STEPS,
):
    """Run a text-to-video benchmark with the given configuration."""
    resolved_display_name = resolve_display_name(
        request=request, fallback=model_info_name
    )
    ttnn_perf_metrics_output_file = f"tt_xla_{resolved_display_name}_perf_metrics"

    print(f"Running video-gen benchmark for model: {model_info_name}")
    print(
        f"""Configuration:
    optimization_level={optimization_level}
    trace_enabled={trace_enabled}
    prompt={prompt!r}
    num_inference_steps={num_inference_steps}
    height={height}
    width={width}
    num_frames={num_frames}
    ttnn_perf_metrics_output_file={ttnn_perf_metrics_output_file}
    """
    )

    results = benchmark_video_gen_pipeline_torch_xla(
        build_pipeline_fn=build_pipeline_fn,
        model_info_name=model_info_name,
        display_name=resolved_display_name,
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        num_frames=num_frames,
        fps=fps,
        optimization_level=optimization_level,
        trace_enabled=trace_enabled,
        ttnn_perf_metrics_output_file=ttnn_perf_metrics_output_file,
        output_video_path=output_video_path,
        warmup_steps=warmup_steps,
    )

    if output_file:
        results["project"] = "tt-forge/tt-xla"
        results["model_rawname"] = model_info_name

        aggregate_ttnn_perf_metrics(ttnn_perf_metrics_output_file, results)

        with open(output_file, "w") as file:
            json.dump(results, file, indent=2)


def test_hunyuan_video_1_5(output_file, request):
    """HunyuanVideo 1.5 (480p t2v base): DiT and text encoders on TT,
    scheduler/guider combine/VAE on CPU. Config matches the nightly pipeline
    test. Guidance is real CFG, so each step is two DiT forwards and one
    `transformer_step` entry covers both."""
    from third_party.tt_forge_models.hunyuan_1_5.pytorch.src.pipeline import (
        HunyuanVideo15Config,
        HunyuanVideo15Pipeline,
    )

    # The double-quoted span is what routes text through text_encoder_2 (the
    # ByT5 glyph encoder); without it that stage never runs.
    PROMPT = 'A girl holding a paper with words "Hello, world!"'
    HEIGHT = 480
    WIDTH = 848
    NUM_FRAMES = 25
    NUM_INFERENCE_STEPS = 10
    FPS = 15

    def build_pipeline_fn(compile_options):
        # compile_options are applied globally by the harness before this call;
        # the pipeline configures TT via SPMD sharding and needs nothing further.
        pipeline = HunyuanVideo15Pipeline(
            config=HunyuanVideo15Config(
                num_inference_steps=NUM_INFERENCE_STEPS,
                height=HEIGHT,
                width=WIDTH,
                num_frames=NUM_FRAMES,
            )
        )
        pipeline.setup()

        def generate_fn(prompt, num_inference_steps):
            return pipeline.generate(
                prompt=prompt,
                seed=DEFAULT_SEED,
                num_inference_steps=num_inference_steps,
                output_type="pil",
            )

        return pipeline, generate_fn

    _run_video_gen(
        build_pipeline_fn=build_pipeline_fn,
        model_info_name="HunyuanVideo-1.5-480p-t2v",
        output_file=output_file,
        prompt=PROMPT,
        num_inference_steps=NUM_INFERENCE_STEPS,
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        fps=FPS,
        request=request,
    )


def test_mochi_1_preview(output_file, request):
    """Mochi-1 preview: DiT and T5-XXL text encoder on TT, scheduler/VAE on CPU.
    Config matches the nightly pipeline test."""
    from third_party.tt_forge_models.mochi.pytorch.src.pipeline import (
        Mochi1Config,
        Mochi1Pipeline,
    )

    PROMPT = (
        "Close-up of a chameleon's eye, with its scaly skin changing color. "
        "Ultra high resolution 4k."
    )
    HEIGHT = 480
    WIDTH = 848
    # 24 frames / 10 steps instead of the stock 84 / 64:
    # https://github.com/tenstorrent/tt-xla/issues/4638
    NUM_FRAMES = 24
    NUM_INFERENCE_STEPS = 10
    FPS = 15
    # Mochi's linear_quadratic_schedule splits the steps into a linear and a
    # quadratic half, so at 1 step linear_steps == 0 and it divides by zero.
    # 2 is the smallest workable value; the compiled graph is the same either
    # way. https://github.com/tenstorrent/tt-xla/issues/5999
    WARMUP_STEPS = 2

    def build_pipeline_fn(compile_options):
        # compile_options are applied globally by the harness before this call;
        # the pipeline configures TT via SPMD sharding and needs nothing further.
        pipeline = Mochi1Pipeline(
            config=Mochi1Config(
                num_inference_steps=NUM_INFERENCE_STEPS,
                height=HEIGHT,
                width=WIDTH,
                num_frames=NUM_FRAMES,
            )
        )
        pipeline.setup()

        def generate_fn(prompt, num_inference_steps):
            return pipeline.generate(
                prompt=prompt,
                seed=DEFAULT_SEED,
                num_inference_steps=num_inference_steps,
                output_type="pil",
            )

        return pipeline, generate_fn

    _run_video_gen(
        build_pipeline_fn=build_pipeline_fn,
        model_info_name="Mochi-1-preview",
        output_file=output_file,
        prompt=PROMPT,
        num_inference_steps=NUM_INFERENCE_STEPS,
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        fps=FPS,
        request=request,
        warmup_steps=WARMUP_STEPS,
    )
