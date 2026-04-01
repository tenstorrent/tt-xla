# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from benchmarks.vllm_benchmark import VLLMBenchmarkConfig, benchmark_vllm
from utils import resolve_display_name

SINGLE_DEVICE_CONFIGS = [
    pytest.param(
        VLLMBenchmarkConfig(
            model="meta-llama/Llama-3.2-1B",
            batch_size=1,
            max_model_len=128,
        ),
        id="llama-3.2-1b",
    ),
    pytest.param(
        VLLMBenchmarkConfig(
            model="meta-llama/Llama-3.2-1B",
            batch_size=16,
            max_model_len=128,
            gpu_memory_utilization=0.037,
        ),
        id="llama-3.2-1b-batch16",
    ),
    pytest.param(
        VLLMBenchmarkConfig(
            model="meta-llama/Llama-3.2-1B",
            batch_size=32,
            max_model_len=128,
            gpu_memory_utilization=0.037,
        ),
        id="llama-3.2-1b-batch32",
    ),
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
        VLLMBenchmarkConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            batch_size=1,
            max_model_len=128,
        ),
        id="llama-3.1-8b",
    ),
    pytest.param(
        VLLMBenchmarkConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            batch_size=16,
            max_model_len=128,
            gpu_memory_utilization=0.037,
        ),
        id="llama-3.1-8b-batch16",
    ),
    pytest.param(
        VLLMBenchmarkConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            batch_size=32,
            max_model_len=128,
            gpu_memory_utilization=0.037,
        ),
        id="llama-3.1-8b-batch32",
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


# Batch scaling sweep: 4 models x 3 batch sizes x 2 sampling modes (on-device).
# Matches perf_debug/run_batch_scaling_sweep.sh for comparison.
# Requires fix: fa29afe28 (multi-core topk for non-greedy sampling).
BATCH_SCALING_CONFIGS = [
    # OPT-125M
    pytest.param(
        VLLMBenchmarkConfig(
            model="facebook/opt-125m",
            batch_size=1,
            max_model_len=128,
            cpu_sampling=False,
            temperature=0.0,
        ),
        id="opt-125m-batch1-greedy",
    ),
    pytest.param(
        VLLMBenchmarkConfig(
            model="facebook/opt-125m",
            batch_size=1,
            max_model_len=128,
            cpu_sampling=False,
            temperature=0.8,
            top_p=0.9,
        ),
        id="opt-125m-batch1-non-greedy",
    ),
    pytest.param(
        VLLMBenchmarkConfig(
            model="facebook/opt-125m",
            batch_size=16,
            max_model_len=128,
            cpu_sampling=False,
            temperature=0.0,
            gpu_memory_utilization=0.037,
        ),
        id="opt-125m-batch16-greedy",
    ),
    pytest.param(
        VLLMBenchmarkConfig(
            model="facebook/opt-125m",
            batch_size=16,
            max_model_len=128,
            cpu_sampling=False,
            temperature=0.8,
            top_p=0.9,
            gpu_memory_utilization=0.037,
        ),
        id="opt-125m-batch16-non-greedy",
    ),
    pytest.param(
        VLLMBenchmarkConfig(
            model="facebook/opt-125m",
            batch_size=32,
            max_model_len=128,
            cpu_sampling=False,
            temperature=0.0,
            gpu_memory_utilization=0.037,
        ),
        id="opt-125m-batch32-greedy",
    ),
    pytest.param(
        VLLMBenchmarkConfig(
            model="facebook/opt-125m",
            batch_size=32,
            max_model_len=128,
            cpu_sampling=False,
            temperature=0.8,
            top_p=0.9,
            gpu_memory_utilization=0.037,
        ),
        id="opt-125m-batch32-non-greedy",
    ),
    # Llama-3.2-1B-Instruct
    pytest.param(
        VLLMBenchmarkConfig(
            model="meta-llama/Llama-3.2-1B-Instruct",
            batch_size=1,
            max_model_len=128,
            cpu_sampling=False,
            temperature=0.0,
        ),
        id="llama-3.2-1b-instruct-batch1-greedy",
    ),
    pytest.param(
        VLLMBenchmarkConfig(
            model="meta-llama/Llama-3.2-1B-Instruct",
            batch_size=1,
            max_model_len=128,
            cpu_sampling=False,
            temperature=0.8,
            top_p=0.9,
        ),
        id="llama-3.2-1b-instruct-batch1-non-greedy",
    ),
    pytest.param(
        VLLMBenchmarkConfig(
            model="meta-llama/Llama-3.2-1B-Instruct",
            batch_size=16,
            max_model_len=128,
            cpu_sampling=False,
            temperature=0.0,
            gpu_memory_utilization=0.037,
        ),
        id="llama-3.2-1b-instruct-batch16-greedy",
    ),
    pytest.param(
        VLLMBenchmarkConfig(
            model="meta-llama/Llama-3.2-1B-Instruct",
            batch_size=16,
            max_model_len=128,
            cpu_sampling=False,
            temperature=0.8,
            top_p=0.9,
            gpu_memory_utilization=0.037,
        ),
        id="llama-3.2-1b-instruct-batch16-non-greedy",
    ),
    pytest.param(
        VLLMBenchmarkConfig(
            model="meta-llama/Llama-3.2-1B-Instruct",
            batch_size=32,
            max_model_len=128,
            cpu_sampling=False,
            temperature=0.0,
            gpu_memory_utilization=0.037,
        ),
        id="llama-3.2-1b-instruct-batch32-greedy",
    ),
    pytest.param(
        VLLMBenchmarkConfig(
            model="meta-llama/Llama-3.2-1B-Instruct",
            batch_size=32,
            max_model_len=128,
            cpu_sampling=False,
            temperature=0.8,
            top_p=0.9,
            gpu_memory_utilization=0.037,
        ),
        id="llama-3.2-1b-instruct-batch32-non-greedy",
    ),
    # Llama-3.2-3B-Instruct
    pytest.param(
        VLLMBenchmarkConfig(
            model="meta-llama/Llama-3.2-3B-Instruct",
            batch_size=1,
            max_model_len=128,
            cpu_sampling=False,
            temperature=0.0,
        ),
        id="llama-3.2-3b-instruct-batch1-greedy",
    ),
    pytest.param(
        VLLMBenchmarkConfig(
            model="meta-llama/Llama-3.2-3B-Instruct",
            batch_size=1,
            max_model_len=128,
            cpu_sampling=False,
            temperature=0.8,
            top_p=0.9,
        ),
        id="llama-3.2-3b-instruct-batch1-non-greedy",
    ),
    pytest.param(
        VLLMBenchmarkConfig(
            model="meta-llama/Llama-3.2-3B-Instruct",
            batch_size=16,
            max_model_len=128,
            cpu_sampling=False,
            temperature=0.0,
            gpu_memory_utilization=0.037,
        ),
        id="llama-3.2-3b-instruct-batch16-greedy",
    ),
    pytest.param(
        VLLMBenchmarkConfig(
            model="meta-llama/Llama-3.2-3B-Instruct",
            batch_size=16,
            max_model_len=128,
            cpu_sampling=False,
            temperature=0.8,
            top_p=0.9,
            gpu_memory_utilization=0.037,
        ),
        id="llama-3.2-3b-instruct-batch16-non-greedy",
    ),
    pytest.param(
        VLLMBenchmarkConfig(
            model="meta-llama/Llama-3.2-3B-Instruct",
            batch_size=32,
            max_model_len=128,
            cpu_sampling=False,
            temperature=0.0,
            gpu_memory_utilization=0.037,
        ),
        id="llama-3.2-3b-instruct-batch32-greedy",
    ),
    pytest.param(
        VLLMBenchmarkConfig(
            model="meta-llama/Llama-3.2-3B-Instruct",
            batch_size=32,
            max_model_len=128,
            cpu_sampling=False,
            temperature=0.8,
            top_p=0.9,
            gpu_memory_utilization=0.037,
        ),
        id="llama-3.2-3b-instruct-batch32-non-greedy",
    ),
    # Llama-3.1-8B-Instruct
    pytest.param(
        VLLMBenchmarkConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            batch_size=1,
            max_model_len=128,
            cpu_sampling=False,
            temperature=0.0,
        ),
        id="llama-3.1-8b-instruct-batch1-greedy",
    ),
    pytest.param(
        VLLMBenchmarkConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            batch_size=1,
            max_model_len=128,
            cpu_sampling=False,
            temperature=0.8,
            top_p=0.9,
        ),
        id="llama-3.1-8b-instruct-batch1-non-greedy",
    ),
    pytest.param(
        VLLMBenchmarkConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            batch_size=16,
            max_model_len=128,
            cpu_sampling=False,
            temperature=0.0,
            gpu_memory_utilization=0.037,
        ),
        id="llama-3.1-8b-instruct-batch16-greedy",
    ),
    pytest.param(
        VLLMBenchmarkConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            batch_size=16,
            max_model_len=128,
            cpu_sampling=False,
            temperature=0.8,
            top_p=0.9,
            gpu_memory_utilization=0.037,
        ),
        id="llama-3.1-8b-instruct-batch16-non-greedy",
    ),
    pytest.param(
        VLLMBenchmarkConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            batch_size=32,
            max_model_len=128,
            cpu_sampling=False,
            temperature=0.0,
            gpu_memory_utilization=0.037,
        ),
        id="llama-3.1-8b-instruct-batch32-greedy",
    ),
    pytest.param(
        VLLMBenchmarkConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            batch_size=32,
            max_model_len=128,
            cpu_sampling=False,
            temperature=0.8,
            top_p=0.9,
            gpu_memory_utilization=0.037,
        ),
        id="llama-3.1-8b-instruct-batch32-non-greedy",
    ),
]


@pytest.mark.parametrize("config", BATCH_SCALING_CONFIGS)
def test_vllm_batch_scaling(config, output_file, request):
    _run_vllm_benchmark(config, output_file, request)
