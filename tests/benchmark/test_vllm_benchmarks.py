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


# Trace comparison: greedy device sampling × trace on/off
_OPT_BASE = dict(
    model="facebook/opt-125m",
    batch_size=1,
    max_model_len=128,
    gpu_memory_utilization=0.001,
)
_OPT_COMMON = dict(enable_const_eval=False, min_context_len=32, cpu_sampling=False)

_LLAMA_1B_BASE = dict(
    model="meta-llama/Llama-3.2-1B",
    batch_size=1,
    max_model_len=128,
    gpu_memory_utilization=0.002,
)
_LLAMA_1B_COMMON = dict(enable_const_eval=False, min_context_len=32, cpu_sampling=False)

_LLAMA_8B_BASE = dict(
    model="meta-llama/Llama-3.1-8B-Instruct",
    batch_size=1,
    max_model_len=128,
    gpu_memory_utilization=0.05,
)
_LLAMA_8B_COMMON = dict(
    enable_const_eval=True,
    cpu_sampling=False,
    experimental_weight_dtype="bfp_bf8",
    optimization_level=1,
)

TRACE_CONFIGS = [
    # OPT-125m (no RoPE — trace works)
    pytest.param(
        VLLMBenchmarkConfig(
            **_OPT_BASE, additional_config={**_OPT_COMMON, "enable_trace": False}
        ),
        id="opt-125m-greedy-notrace",
    ),
    pytest.param(
        VLLMBenchmarkConfig(
            **_OPT_BASE, additional_config={**_OPT_COMMON, "enable_trace": True}
        ),
        id="opt-125m-greedy-trace",
    ),
    # Llama-3.2-1B
    pytest.param(
        VLLMBenchmarkConfig(
            **_LLAMA_1B_BASE,
            additional_config={**_LLAMA_1B_COMMON, "enable_trace": False},
        ),
        id="llama-1b-greedy-notrace",
    ),
    pytest.param(
        VLLMBenchmarkConfig(
            **_LLAMA_1B_BASE,
            additional_config={**_LLAMA_1B_COMMON, "enable_trace": True},
        ),
        id="llama-1b-greedy-trace",
    ),
    # Llama-3.1-8B-Instruct (greedy only, bfp8)
    pytest.param(
        VLLMBenchmarkConfig(
            **_LLAMA_8B_BASE,
            additional_config={**_LLAMA_8B_COMMON, "enable_trace": False},
        ),
        id="llama-8b-greedy-notrace",
    ),
    pytest.param(
        VLLMBenchmarkConfig(
            **_LLAMA_8B_BASE,
            additional_config={**_LLAMA_8B_COMMON, "enable_trace": True},
        ),
        id="llama-8b-greedy-trace",
    ),
]


@pytest.mark.parametrize("config", TRACE_CONFIGS)
def test_vllm_trace(config, output_file, request):
    _run_vllm_benchmark(config, output_file, request)
