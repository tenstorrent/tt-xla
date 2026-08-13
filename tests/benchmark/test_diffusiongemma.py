# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Perf benchmark for DiffusionGemma-26B-A4B-it block-diffusion text generation.

Unlike the autoregressive models in ``test_llms.py`` there is no per-token decode loop:
generation denoises a whole canvas block, so throughput is generated-tokens / wall-clock
over one ``manual_generate`` call and there is no TTFT.

There is no warmup-then-time loop like the other benchmarks here. Encoder and decoder cannot
be co-resident, so ``free_tt_graphs()`` runs between stages and every call recompiles. That
is not just the double-load described in tenstorrent/tt-xla#5538 -- the encoder's text
weights are in fact tied to the decoder's, so one load covers both -- but the shard map
replicates attention (2 global KV heads can't shard 8 devices) and the embeddings, so a
single co-resident copy still exhausts device DRAM (measured: OOM placing lm_head, 1.48GB,
with ~91GB of ~98GB already allocated).

Two numbers are therefore reported:

  * ``samples_per_sec`` -- generated-tokens / wall-clock for the whole call, compile and
    weight loading included. This is what a user actually waits for.
  * ``warm_decode_steps_per_sec`` -- mean over denoising steps 2..N, which reuse the decoder
    graph compiled on step 1. This is the compile-free number comparable to the warm
    measurements the other benchmarks report.
"""

import inspect
import json
import time

import pytest
import torch
import torch_xla.runtime as xr
from utils import (
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
DEFAULT_BATCH_SIZE = 1
MODEL_INFO_NAME = "google/diffusiongemma-26B-A4B-it"


@pytest.mark.nightly
@pytest.mark.llmbox
def test_diffusiongemma_26b(
    output_file,
    request,
    loop_count=DEFAULT_LOOP_COUNT,
    data_format=DEFAULT_DATA_FORMAT,
    batch_size=DEFAULT_BATCH_SIZE,
):
    """End-to-end text-generation throughput for DiffusionGemma 26B on 8 chips."""
    from third_party.tt_forge_models.diffusiongemma.pytorch import (
        loader as diffgemma_loader,
    )

    xr.set_device_type("TT")
    resolved_display_name = resolve_display_name(
        request=request, fallback="diffusiongemma_26b_a4b_it"
    )

    # transformers>=5.11 is required for DiffusionGemma but the env is pinned lower;
    # install the loader's version for this run and roll back on exit.
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

        setup_time = time.perf_counter() - setup_start

        # manual_generate is driven directly (rather than pipeline.generate) so the
        # generated token count is exact instead of inferred from decoded text.
        total_time = 0.0
        total_new_tokens = 0
        warm_step_times = []
        for _ in range(loop_count):
            encoder_forward, decoder_forward = pipeline._staged_forwards(vocab_size)
            step_times = []

            # Step 1 compiles the decoder; 2..N reuse that graph, so only those are warm.
            def timed_decoder_forward(_inner=decoder_forward, _acc=step_times, **kw):
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
            warm_step_times.extend(step_times[1:])

    tokens_per_sec = total_new_tokens / total_time if total_time else 0.0
    mean_warm_step = (
        sum(warm_step_times) / len(warm_step_times) if warm_step_times else 0.0
    )
    warm_steps_per_sec = 1.0 / mean_warm_step if mean_warm_step else 0.0
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
            create_measurement(
                "warm_decode_steps_per_sec", warm_steps_per_sec, MODEL_INFO_NAME
            ),
            create_measurement(
                "mean_warm_decode_step_s", mean_warm_step, MODEL_INFO_NAME
            ),
            create_measurement(
                "warm_decode_steps", len(warm_step_times), MODEL_INFO_NAME
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
