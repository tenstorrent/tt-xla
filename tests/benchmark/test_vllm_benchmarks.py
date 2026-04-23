# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import os

import pytest
from benchmarks.vllm_benchmark import VLLMBenchmarkConfig, benchmark_vllm
from utils import resolve_display_name

# Set VLLM_ENABLE_TRACE=1 to run the trace benchmark configs with metal trace enabled.
# Default is off so CI measures the stable no-trace baseline.
_ENABLE_TRACE = os.environ.get("VLLM_ENABLE_TRACE", "0").lower() in ("1", "true")

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

# Trace benchmarks: device sampling is required for the metal-trace path, so
# these configs opt out of the cpu_sampling default used by the main benchmark.
# bfp8 weights + const-eval + opt_level=1 matches the production 8B config and
# keeps the decode graph small enough for trace on all three models.
_TRACE_ADDITIONAL_CONFIG = {
    "cpu_sampling": False,
    "enable_trace": _ENABLE_TRACE,
    "enable_const_eval": True,
    "experimental_weight_dtype": "bfp_bf8",
    "optimization_level": 1,
}

TRACE_CONFIGS = [
    pytest.param(
        VLLMBenchmarkConfig(
            model="facebook/opt-125m",
            batch_size=1,
            max_model_len=128,
            gpu_memory_utilization=0.001,
            additional_config=dict(_TRACE_ADDITIONAL_CONFIG),
        ),
        id="opt-125m",
    ),
    pytest.param(
        VLLMBenchmarkConfig(
            model="meta-llama/Llama-3.2-1B",
            batch_size=1,
            max_model_len=128,
            additional_config=dict(_TRACE_ADDITIONAL_CONFIG),
        ),
        id="llama-3.2-1b",
    ),
    pytest.param(
        VLLMBenchmarkConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            batch_size=1,
            max_model_len=128,
            gpu_memory_utilization=0.05,
            additional_config=dict(_TRACE_ADDITIONAL_CONFIG),
        ),
        id="llama-3.1-8b",
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


@pytest.mark.parametrize("config", TRACE_CONFIGS)
def test_vllm_trace_benchmark(config, output_file, request):
    _run_vllm_benchmark(config, output_file, request)
