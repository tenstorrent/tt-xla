# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Perf benchmark for DiffusionGemma-26B-A4B-it block-diffusion text generation.

Generation denoises a whole canvas block, so there is no TTFT in the usual sense and
throughput is generated-tokens / wall-clock.

Encoder and decoder cannot be co-resident, so one is evicted before the other loads -- and
eviction discards the compiled graph. Warm numbers are therefore per component while it is
resident; the encoder repeat lives in ``_staged_forwards``, which frees it on return.
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
)

from tests.runner.requirements import RequirementsManager

DEFAULT_DATA_FORMAT = "bfloat16"
DEFAULT_LOOP_COUNT = 1
DEFAULT_WARM_ENCODER_ITERS = 3
DEFAULT_BATCH_SIZE = 1
MODEL_INFO_NAME = "google/diffusiongemma-26B-A4B-it"
MODULE_EXPORT_PATH = "modules"


@pytest.mark.nightly
@pytest.mark.llmbox
def test_diffusiongemma_26b(
    output_file,
    request,
    loop_count=DEFAULT_LOOP_COUNT,
    warm_encoder_iters=DEFAULT_WARM_ENCODER_ITERS,
    data_format=DEFAULT_DATA_FORMAT,
    batch_size=DEFAULT_BATCH_SIZE,
):
    """End-to-end text generation plus per-component warm timings on 8 chips."""
    from third_party.tt_forge_models.diffusiongemma.pytorch import (
        loader as diffgemma_loader,
    )

    xr.set_device_type("TT")
    resolved_display_name = resolve_display_name(
        request=request, fallback="diffusiongemma_26b_a4b_it"
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
            manual_generate,
        )

        pipeline = DiffusionGemmaPipeline(
            config=DiffusionGemmaConfig(max_new_tokens=MAX_NEW_TOKENS, seed=SEED)
        )
        setup_start = time.perf_counter()
        pipeline.setup()
        setup_time = time.perf_counter() - setup_start

        inputs = pipeline.loader.load_inputs(
            dtype_override=torch.bfloat16, prompt=PROMPT
        )
        extra_kwargs = {
            k: v
            for k, v in inputs.items()
            if k not in ("input_ids", "attention_mask", "decoder_input_ids")
        }
        prompt_len = inputs["input_ids"].shape[-1]
        vocab_size = pipeline.cpu_model.config.text_config.vocab_size

        encoder_times = []
        decode_step_times = []
        total_time = 0.0
        total_new_tokens = 0

        for _ in range(loop_count):
            encoder_forward, decoder_forward = pipeline._staged_forwards(
                vocab_size,
                encoder_iters=max(1, warm_encoder_iters),
                encoder_times=encoder_times,
            )

            # Step 1 builds the graph; 2..N reuse it while resident.
            def timed_decoder_forward(
                _inner=decoder_forward, _acc=decode_step_times, **kw
            ):
                step_start = time.perf_counter()
                out = _inner(**kw)
                _acc.append(time.perf_counter() - step_start)
                return out

            torch.manual_seed(SEED)
            start = time.perf_counter()
            output = manual_generate(
                pipeline.cpu_model,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=MAX_NEW_TOKENS,
                encoder_forward=encoder_forward,
                decoder_forward=timed_decoder_forward,
                **extra_kwargs,
            )
            total_time += time.perf_counter() - start
            total_new_tokens += int(output.shape[-1] - prompt_len)

    tokens_per_sec = total_new_tokens / total_time if total_time else 0.0

    def _mean(values):
        return sum(values) / len(values) if values else 0.0

    # Drop the first of each: it carries the build.
    warm_encoder_s = _mean(encoder_times[1:])
    warm_decode_step_s = _mean(decode_step_times[1:])
    cold_encoder_s = encoder_times[0] if encoder_times else 0.0
    cold_decode_step_s = decode_step_times[0] if decode_step_times else 0.0

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
        model_title="DiffusionGemma 26B-A4B-it",
        full_model_name=MODEL_INFO_NAME,
        model_type="text-generation",
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
        model_type="text-generation",
        dataset_name="na",
        num_layers=-1,
        batch_size=batch_size,
        input_size=(batch_size, prompt_len),
        loop_count=loop_count,
        data_format=data_format,
        total_time=total_time,
        total_samples=total_new_tokens,
        custom_measurements=[
            create_measurement("tokens_per_sec", tokens_per_sec, MODEL_INFO_NAME),
            create_measurement("setup_time", setup_time, MODEL_INFO_NAME),
            create_measurement("max_new_tokens", MAX_NEW_TOKENS, MODEL_INFO_NAME),
            # Measured while that component is resident.
            create_measurement("warm_encoder_s", warm_encoder_s, MODEL_INFO_NAME),
            create_measurement("cold_encoder_s", cold_encoder_s, MODEL_INFO_NAME),
            create_measurement(
                "warm_decode_step_s", warm_decode_step_s, MODEL_INFO_NAME
            ),
            create_measurement(
                "cold_decode_step_s", cold_decode_step_s, MODEL_INFO_NAME
            ),
            create_measurement(
                "warm_decode_steps", max(0, len(decode_step_times) - 1), MODEL_INFO_NAME
            ),
        ],
        display_name=resolved_display_name,
        arch=arch,
        input_is_image=False,
        input_sequence_length=prompt_len,
        device_count=device_count,
        mesh_shape=(1, device_count),
    )

    if output_file:
        results["project"] = "tt-forge/tt-xla"
        results["model_rawname"] = MODEL_INFO_NAME
        with open(output_file, "w") as file:
            json.dump(results, file, indent=2)
