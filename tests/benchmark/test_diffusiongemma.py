# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Perf benchmark for DiffusionGemma-26B-A4B-it block-diffusion text generation.

Generation denoises a whole canvas block, so there is no TTFT in the usual sense and
throughput is generated-tokens / wall-clock.

Encoder and decoder cannot be co-resident, so one is evicted before the other loads -- and
eviction discards the compiled graph. Warm numbers are therefore per component while it is
resident, taken before the eviction.

The pipeline times itself into ``_perf``; this file only calls the shipped ``generate()``
and translates, so it cannot drift from the code the demo and the PCC test run.
"""

import inspect
import json
import time

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from loguru import logger
from utils import (
    build_xla_export_name,
    create_benchmark_result,
    create_measurement,
    get_benchmark_metadata,
    get_xla_device_arch,
    print_benchmark_results,
    resolve_display_name,
    staged_perf_measurements,
)

from tests.runner.requirements import RequirementsManager

DEFAULT_DATA_FORMAT = "bfloat16"
DEFAULT_WARM_ENCODER_ITERS = 2  # extra in-residency prefills
DEFAULT_BATCH_SIZE = 1
MODEL_INFO_NAME = "google/diffusiongemma-26B-A4B-it"
MODULE_EXPORT_PATH = "modules"


def _run_diffusiongemma_benchmark(
    output_file,
    request,
    image,
    fallback_display_name,
    warm_encoder_iters,
    data_format,
    batch_size,
):
    """End-to-end generation plus per-component warm timings on 8 chips.

    ``image`` selects image+text inputs; the model, shard spec and staged residency
    are identical either way, so only the inputs and the reported model type differ.
    """
    from third_party.tt_forge_models.diffusiongemma.pytorch import (
        loader as diffgemma_loader,
    )

    xr.set_device_type("TT")
    resolved_display_name = resolve_display_name(
        request=request, fallback=fallback_display_name
    )

    # The shared benchmarks/ harnesses own this block; this benchmark measures
    # staged residency itself and so never passes through one. Without it no
    # ./modules/irs/*.mlir is written and the CI IR-dump steps fail on a bare cp.
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

    loader_path = inspect.getsourcefile(diffgemma_loader)
    with RequirementsManager.for_loader(loader_path, framework="torch"):
        from third_party.tt_forge_models.diffusiongemma.pytorch.pipeline import (
            MAX_NEW_TOKENS,
            PROMPT,
            SEED,
            DiffusionGemmaConfig,
            DiffusionGemmaPipeline,
        )

        pipeline = DiffusionGemmaPipeline(
            config=DiffusionGemmaConfig(
                max_new_tokens=MAX_NEW_TOKENS,
                seed=SEED,
                warm_iters=warm_encoder_iters,
                image=image,
            )
        )
        setup_start = time.perf_counter()
        pipeline.setup()
        setup_time = time.perf_counter() - setup_start

        prompt_inputs = (
            pipeline.loader.load_image_inputs(dtype_override=torch.bfloat16)
            if image
            else pipeline.loader.load_text_inputs(
                dtype_override=torch.bfloat16, prompt=PROMPT
            )
        )
        prompt_len = prompt_inputs["input_ids"].shape[-1]

        pipeline.generate()

    model_type = "image-text-to-text" if image else "text-generation"
    model_title = "DiffusionGemma 26B-A4B-it" + (" (image+text)" if image else "")

    perf = pipeline._perf
    total_time = perf["total"]
    total_new_tokens = pipeline.last_new_tokens
    decode_step_times = perf["steps"]
    tokens_per_sec = total_new_tokens / total_time if total_time else 0.0

    # Same schema and translation the image-gen harness uses, so this model's keys
    # match every other benchmark.
    derived = staged_perf_measurements(
        perf,
        step_metric="decode_step",
        step_name=MODEL_INFO_NAME,
        staged_residency=getattr(pipeline, "benchmark_staged_residency", False),
    )
    cold_encoder_s = perf["cold"].get("encoder", 0.0)
    warm_encoder_s = perf["warm"].get("encoder", 0.0)
    cold_decode_step_s = derived["cold"].get("decode_step", 0.0)
    warm_decode_step_s = derived["warm"].get("decode_step", 0.0)

    logger.info(
        "[PERF] encoder cold={:.2f}s warm={:.2f}s | decode step cold={:.2f}s warm={:.2f}s ({} warm steps)",
        cold_encoder_s,
        warm_encoder_s,
        cold_decode_step_s,
        warm_decode_step_s,
        max(0, len(decode_step_times) - 1),
    )

    metadata = get_benchmark_metadata()
    arch = get_xla_device_arch()
    device_count = xr.global_runtime_device_count()

    print_benchmark_results(
        model_title=model_title,
        full_model_name=MODEL_INFO_NAME,
        model_type=model_type,
        dataset_name="na",
        date=metadata["date"],
        machine_name=metadata["machine_name"],
        total_time=total_time,
        total_samples=total_new_tokens,
        samples_per_sec=tokens_per_sec,
        batch_size=batch_size,
        data_format=data_format,
        input_size=(batch_size, prompt_len),
        input_sequence_length=prompt_len,
        ttft_ms=cold_encoder_s * 1e3,
    )

    results = create_benchmark_result(
        full_model_name=MODEL_INFO_NAME,
        model_type=model_type,
        dataset_name="na",
        num_layers=-1,
        batch_size=batch_size,
        input_size=(batch_size, prompt_len),
        loop_count=1,
        data_format=data_format,
        total_time=total_time,
        total_samples=total_new_tokens,
        custom_measurements=[
            create_measurement("tokens_per_sec", tokens_per_sec, MODEL_INFO_NAME),
            # The measured wall clock, as the image-gen harnesses publish it.
            create_measurement("e2e_latency", total_time, MODEL_INFO_NAME),
            create_measurement("setup_time", setup_time, MODEL_INFO_NAME),
            create_measurement("max_new_tokens", MAX_NEW_TOKENS, MODEL_INFO_NAME),
            create_measurement(
                "warm_decode_steps", max(0, len(decode_step_times) - 1), MODEL_INFO_NAME
            ),
            # encoder_cold_s / encoder_warm_s / decode_step_* / cpu_overhead_s /
            # staging_overhead_s / synthetic_s / e2e_warm_s / e2e_cold_s -- the
            # same names every other benchmark publishes.
            *derived["measurements"],
        ],
        display_name=resolved_display_name,
        arch=arch,
        input_is_image=image,
        input_sequence_length=prompt_len,
        device_count=device_count,
        mesh_shape=(1, device_count),
    )

    if output_file:
        results["project"] = "tt-forge/tt-xla"
        results["model_rawname"] = MODEL_INFO_NAME
        with open(output_file, "w") as file:
            json.dump(results, file, indent=2)


@pytest.mark.nightly
@pytest.mark.llmbox
def test_diffusiongemma_26b(
    output_file,
    request,
    warm_encoder_iters=DEFAULT_WARM_ENCODER_ITERS,
    data_format=DEFAULT_DATA_FORMAT,
    batch_size=DEFAULT_BATCH_SIZE,
):
    """Text-only block-diffusion generation."""
    _run_diffusiongemma_benchmark(
        output_file,
        request,
        image=False,
        fallback_display_name="diffusiongemma_26b_a4b_it",
        warm_encoder_iters=warm_encoder_iters,
        data_format=data_format,
        batch_size=batch_size,
    )


@pytest.mark.nightly
@pytest.mark.llmbox
def test_diffusiongemma_26b_image(
    output_file,
    request,
    warm_encoder_iters=DEFAULT_WARM_ENCODER_ITERS,
    data_format=DEFAULT_DATA_FORMAT,
    batch_size=DEFAULT_BATCH_SIZE,
):
    """Image+text block-diffusion generation: the prompt carries one image."""
    _run_diffusiongemma_benchmark(
        output_file,
        request,
        image=True,
        fallback_display_name="diffusiongemma_26b_a4b_it_image",
        warm_encoder_iters=warm_encoder_iters,
        data_format=data_format,
        batch_size=batch_size,
    )
