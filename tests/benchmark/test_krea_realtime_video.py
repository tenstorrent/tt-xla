# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Perf benchmark for Krea Realtime Video 14B (autoregressive text-to-video).

The three components (UMT5 encoder, CausalWan DiT, VAE decoder) cannot be
co-resident, so each is evicted before the next is placed -- and eviction
discards the compiled graph. Warm numbers are therefore per component while it is
resident, taken before the eviction, so cold and warm both come from the single
``generate()`` pass (staged residency); a second call would recompile from cold.

The pipeline times itself into ``_perf``; this file only calls the shipped
``generate()`` and translates, so it cannot drift from the code the demo and the
PCC test run. NUM_BLOCKS is pinned to 1: >1 is blocked by the block-1 recompute
OOM (https://github.com/tenstorrent/tt-xla/issues/6090).
"""

import json
import time

import pytest
import torch_xla
import torch_xla.runtime as xr
from loguru import logger
from utils import (
    build_xla_export_name,
    create_benchmark_result,
    create_measurement,
    format_staged_perf_summary,
    get_benchmark_metadata,
    get_xla_device_arch,
    print_benchmark_results,
    resolve_display_name,
    staged_perf_measurements,
)

DEFAULT_DATA_FORMAT = "bfloat16"
DEFAULT_WARM_ITERS = 2  # extra in-residency forwards for the single-shot enc/dec
DEFAULT_BATCH_SIZE = 1
NUM_BLOCKS = 1  # >1 blocked by the block-1 recompute OOM (tt-xla#6090)
MODEL_INFO_NAME = "krea/krea-realtime-video"
MODULE_EXPORT_PATH = "modules"
STEP_METRIC = "denoise_step"


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.llmbox
def test_krea_realtime_video_14b(
    output_file,
    request,
    warm_iters=DEFAULT_WARM_ITERS,
    data_format=DEFAULT_DATA_FORMAT,
    batch_size=DEFAULT_BATCH_SIZE,
):
    """End-to-end text-to-video plus per-component warm timings (staged residency)."""
    from third_party.tt_forge_models.krea_realtime_video.pytorch.pipeline import (
        HEIGHT,
        NUM_INFERENCE_STEPS,
        PROMPT,
        SEED,
        WIDTH,
        KreaRealtimePipeline,
    )

    xr.set_device_type("TT")
    resolved_display_name = resolve_display_name(
        request=request, fallback="krea_realtime_video_14b"
    )

    # This benchmark measures staged residency itself and never passes through the
    # shared benchmarks/ harnesses, so it sets the IR-dump export options here;
    # otherwise no ./modules/irs/*.mlir is written and the CI IR-dump step fails.
    torch_xla.set_custom_compile_options(
        {
            "export_path": MODULE_EXPORT_PATH,
            "export_model_name": build_xla_export_name(
                model_name=resolved_display_name,
                num_layers=None,
                batch_size=batch_size,
                input_sequence_length=None,
            ),
        }
    )

    pipeline = KreaRealtimePipeline(warm_iters=warm_iters)
    setup_start = time.perf_counter()
    pipeline.setup()
    setup_time = time.perf_counter() - setup_start

    frames = pipeline.generate(
        prompt=PROMPT,
        num_blocks=NUM_BLOCKS,
        num_inference_steps=NUM_INFERENCE_STEPS,
        seed=SEED,
    )

    perf = pipeline._perf
    total_time = perf["total"]
    num_frames = len(frames)
    # Throughput as generated frames per second (distinct from the mp4 playback fps).
    generated_frames_per_second = num_frames / total_time if total_time else 0.0

    # Same schema and translation the image/video-gen harnesses use, so this
    # model's metric keys match every other benchmark.
    derived = staged_perf_measurements(
        perf,
        step_metric=STEP_METRIC,
        step_name=MODEL_INFO_NAME,
        staged_residency=getattr(pipeline, "benchmark_staged_residency", False),
    )
    cold_encoder_s = perf["cold"].get("encoder", 0.0)
    warm_encoder_s = perf["warm"].get("encoder", 0.0)
    cold_decode_s = perf["cold"].get("vae_decode", 0.0)
    warm_decode_s = perf["warm"].get("vae_decode", 0.0)
    cold_step_s = derived["cold"].get(STEP_METRIC, 0.0)
    warm_step_s = derived["warm"].get(STEP_METRIC, 0.0)

    logger.info(
        "[PERF] encoder cold={:.2f}s warm={:.2f}s | denoise step cold={:.2f}s "
        "warm={:.2f}s ({} warm steps) | vae_decode cold={:.2f}s warm={:.2f}s | "
        "{:.3f} frames/s",
        cold_encoder_s,
        warm_encoder_s,
        cold_step_s,
        warm_step_s,
        max(0, len(perf["steps"]) - 1),
        cold_decode_s,
        warm_decode_s,
        generated_frames_per_second,
    )

    metadata = get_benchmark_metadata()
    arch = get_xla_device_arch()
    device_count = xr.global_runtime_device_count()
    model_type = "Video Generation, Text-to-Video"
    dataset_name = "Text Prompt"
    input_size = (num_frames, 3, HEIGHT, WIDTH)

    print_benchmark_results(
        model_title="Krea Realtime Video 14B",
        full_model_name=MODEL_INFO_NAME,
        model_type=model_type,
        dataset_name=dataset_name,
        date=metadata["date"],
        machine_name=metadata["machine_name"],
        total_time=total_time,
        total_samples=num_frames,
        samples_per_sec=generated_frames_per_second,
        batch_size=batch_size,
        data_format=data_format,
        input_size=input_size,
    )
    print(
        f"| Num inference steps: {NUM_INFERENCE_STEPS}\n"
        f"| Num blocks: {NUM_BLOCKS}\n"
        f"| Num frames: {num_frames}\n"
        f"| Per component, warm (cold in brackets):\n"
        f"{format_staged_perf_summary(derived, STEP_METRIC)}"
    )

    results = create_benchmark_result(
        full_model_name=MODEL_INFO_NAME,
        model_type=model_type,
        dataset_name=dataset_name,
        num_layers=-1,
        batch_size=batch_size,
        input_size=input_size,
        loop_count=NUM_INFERENCE_STEPS,
        data_format=data_format,
        total_time=total_time,
        total_samples=num_frames,
        custom_measurements=[
            create_measurement(
                "generated_frames_per_second",
                generated_frames_per_second,
                MODEL_INFO_NAME,
            ),
            # The measured wall clock; the comparable pair is e2e_warm_s / e2e_cold_s.
            create_measurement("e2e_latency", total_time, MODEL_INFO_NAME),
            create_measurement("setup_time", setup_time, MODEL_INFO_NAME),
            create_measurement("num_frames", num_frames, MODEL_INFO_NAME),
            create_measurement("num_blocks", NUM_BLOCKS, MODEL_INFO_NAME),
            # encoder_cold_s / encoder_warm_s / denoise_step_* / vae_decode_* /
            # cpu_overhead_s / staging_overhead_s / synthetic_s / e2e_warm_s /
            # e2e_cold_s -- the same names every other benchmark publishes.
            *derived["measurements"],
        ],
        display_name=resolved_display_name,
        arch=arch,
        input_is_image=False,
        device_count=device_count,
        mesh_shape=getattr(pipeline, "mesh_shape", None),
    )

    if output_file:
        results["project"] = "tt-forge/tt-xla"
        results["model_rawname"] = MODEL_INFO_NAME
        with open(output_file, "w") as file:
            json.dump(results, file, indent=2)
