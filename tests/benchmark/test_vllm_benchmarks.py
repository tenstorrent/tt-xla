# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from benchmarks.vllm_benchmark import VLLMBenchmarkConfig, benchmark_vllm
from utils import resolve_display_name

# Gemma-4 multimodal forces max_num_batched_tokens >= 2496 (video:
# _VIDEO_MAX_FRAMES=32 * (_VIDEO_MAX_SOFT_TOKENS=70 + 2 + 6) = 2496 in
# vllm/model_executor/models/gemma4_mm.py).
_GEMMA4_MIN_BATCHED_TOKENS = 2560


def _gemma4_e4b_config(
    batch_size: int = 1,
    max_model_len: int = 512,
    gpu_memory_utilization: float = 0.1,
    max_tokens: int = 256,
) -> VLLMBenchmarkConfig:
    # Matches test_generation_single_device_multimodal_e4b: E4B is a
    # text-only sanity run on a single device, no tensor parallelism.
    return VLLMBenchmarkConfig(
        model="google/gemma-4-E4B-it",
        batch_size=batch_size,
        max_model_len=max_model_len,
        max_num_batched_tokens=max(
            batch_size * max_model_len, _GEMMA4_MIN_BATCHED_TOKENS
        ),
        gpu_memory_utilization=gpu_memory_utilization,
        max_tokens=max_tokens,
        additional_config={
            "enable_const_eval": True,
            "min_context_len": 32,
            "enable_tensor_parallel": False,
            "cpu_sampling": False,
        },
    )


SINGLE_DEVICE_CONFIGS = [
    pytest.param(
        VLLMBenchmarkConfig(
            model="meta-llama/Llama-3.2-3B",
            batch_size=1,
            max_model_len=128,
        ),
        id="llama-3.2-3b",
    ),
    pytest.param(
        VLLMBenchmarkConfig(
            model="meta-llama/Llama-3.2-3B",
            batch_size=32,
            max_model_len=128,
            gpu_memory_utilization=0.037,
        ),
        id="llama-3.2-3b-batch32",
    ),
    pytest.param(
        _gemma4_e4b_config(),
        id="gemma-4-e4b-it-b1",
    ),
    pytest.param(
        _gemma4_e4b_config(batch_size=32),
        id="gemma-4-e4b-it-b32",
    ),
]

_GEMMA4_BASE_ADDITIONAL_CONFIG = {
    "enable_const_eval": True,
    "min_context_len": 32,
    "enable_tensor_parallel": True,
    "use_2d_mesh": False,
    "cpu_sampling": False,
    "experimental_enable_permute_matmul_fusion": False,
    "experimental_weight_dtype": "bfp_bf8",
}


def _gemma4_bhqb_config(
    model: str,
    batch_size: int = 1,
    max_model_len: int = 512,
    max_num_batched_tokens: int = _GEMMA4_MIN_BATCHED_TOKENS,
    gpu_memory_utilization: float = 0.1,
    max_tokens: int = 256,
) -> VLLMBenchmarkConfig:
    additional_config = dict(_GEMMA4_BASE_ADDITIONAL_CONFIG)
    return VLLMBenchmarkConfig(
        model=model,
        batch_size=batch_size,
        max_model_len=max_model_len,
        # tt-xla's model_runner asserts
        #   max_num_batched_tokens >= max_model_len * max_num_seqs,
        # so any value below that would fail engine init. The benchmark keeps
        # this capped to _GEMMA4_MIN_BATCHED_TOKENS (multimodal minimum) for
        # the b1 case, and widens it only when the caller explicitly sizes for
        # larger batches.
        max_num_batched_tokens=max_num_batched_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        max_tokens=max_tokens,
        # QB2 blackhole 4-chip, 1x4 mesh (follows PR #4212 naming).
        arch="qb2-blackhole",
        device_count=4,
        mesh_shape=(1, 4),
        additional_config=additional_config,
    )


BHQB_CONFIGS = [
    pytest.param(
        _gemma4_bhqb_config(model="google/gemma-4-31B-it"),
        id="gemma-4-31b-it-b1",
    ),
    pytest.param(
        _gemma4_bhqb_config(
            model="google/gemma-4-31B-it",
            batch_size=32,
            max_model_len=256,
            max_num_batched_tokens=8192,
            gpu_memory_utilization=0.74,
        ),
        id="gemma-4-31b-it-b32",
    ),
]


def _run_vllm_benchmark(config, output_file, request):
    display_name = "vllm_" + resolve_display_name(
        request=request, fallback=config.model
    )

    print(f"\n{'='*60}")
    print(f"vLLM Benchmark: {display_name}")
    print(f"{'='*60}")

    results = benchmark_vllm(config, display_name)

    if output_file:
        results["project"] = "tt-xla"
        results["model_rawname"] = config.model
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to {output_file}")


@pytest.mark.parametrize("config", SINGLE_DEVICE_CONFIGS)
def test_vllm_benchmark(config, output_file, request):
    _run_vllm_benchmark(config, output_file, request)


@pytest.mark.bhqb
@pytest.mark.tensor_parallel
@pytest.mark.parametrize("config", BHQB_CONFIGS)
def test_vllm_benchmark_bhqb(config, output_file, request):
    _run_vllm_benchmark(config, output_file, request)
