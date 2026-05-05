# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import os

import pytest
from benchmarks.vllm_benchmark import VLLMBenchmarkConfig, benchmark_vllm
from utils import resolve_display_name

# Sampling overrides — keep SINGLE_DEVICE_CONFIGS focused on (model,
# batch_size). CI re-runs the same matrix with different sampling
# configs by setting these env vars (one knob per re-run):
#   TT_BENCHMARK_TEMPERATURE=<float>  default 0.0 (greedy)
#   TT_BENCHMARK_CPU_SAMPLING=1       default 0 (device sampling)
_BENCH_TEMPERATURE = float(os.environ.get("TT_BENCHMARK_TEMPERATURE", "0.0"))
_BENCH_CPU_SAMPLING = os.environ.get("TT_BENCHMARK_CPU_SAMPLING", "0") == "1"


def _config(model: str, batch_size: int, *, gpu_memory_utilization: float = 0.05):
    additional = {"enable_trace": True}
    if _BENCH_CPU_SAMPLING:
        additional["cpu_sampling"] = True
    return VLLMBenchmarkConfig(
        model=model,
        batch_size=batch_size,
        max_model_len=128,
        gpu_memory_utilization=gpu_memory_utilization,
        temperature=_BENCH_TEMPERATURE,
        additional_config=additional,
    )


SINGLE_DEVICE_CONFIGS = [
    pytest.param(_config("meta-llama/Llama-3.2-3B", 1), id="llama-3.2-3b"),
    pytest.param(
        _config("meta-llama/Llama-3.2-3B", 32, gpu_memory_utilization=0.037),
        id="llama-3.2-3b-batch32",
    ),
    pytest.param(_config("meta-llama/Llama-3.2-1B-Instruct", 1), id="llama-3.2-1b"),
    pytest.param(
        _config("meta-llama/Llama-3.2-1B-Instruct", 32),
        id="llama-3.2-1b-batch32",
    ),
    pytest.param(_config("meta-llama/Llama-3.1-8B-Instruct", 1), id="llama-3.1-8b"),
    pytest.param(
        _config("meta-llama/Llama-3.1-8B-Instruct", 32),
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
