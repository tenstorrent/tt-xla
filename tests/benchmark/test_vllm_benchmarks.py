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
            gpu_memory_utilization=0.05,
        ),
        id="llama-3.2-3b",
    ),
    pytest.param(
        VLLMBenchmarkConfig(
            model="meta-llama/Llama-3.2-3B",
            batch_size=32,
            max_model_len=128,
            gpu_memory_utilization=0.05,
        ),
        id="llama-3.2-3b-batch32",
    ),
    pytest.param(
        VLLMBenchmarkConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            batch_size=1,
            max_model_len=128,
            gpu_memory_utilization=0.05,
            additional_config={
                "enable_const_eval": True,
                "cpu_sampling": False,
                "experimental_weight_dtype": "bfp_bf8",
                "optimization_level": 1,
            },
        ),
        id="llama-3.1-8b-instruct",
    ),
    pytest.param(
        VLLMBenchmarkConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            batch_size=32,
            max_model_len=128,
            gpu_memory_utilization=0.05,
            additional_config={
                "enable_const_eval": True,
                "cpu_sampling": False,
                "experimental_weight_dtype": "bfp_bf8",
                "optimization_level": 1,
            },
        ),
        id="llama-3.1-8b-instruct-batch32",
    ),
    pytest.param(
        VLLMBenchmarkConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            batch_size=2,
            max_model_len=128,
            gpu_memory_utilization=0.05,
            additional_config={
                "enable_const_eval": True,
                "cpu_sampling": False,
                "experimental_weight_dtype": "bfp_bf8",
                "optimization_level": 1,
            },
        ),
        id="llama-3.1-8b-instruct-batch2",
    ),
    pytest.param(
        VLLMBenchmarkConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            batch_size=4,
            max_model_len=128,
            gpu_memory_utilization=0.05,
            additional_config={
                "enable_const_eval": True,
                "cpu_sampling": False,
                "experimental_weight_dtype": "bfp_bf8",
                "optimization_level": 1,
            },
        ),
        id="llama-3.1-8b-instruct-batch4",
    ),
    pytest.param(
        VLLMBenchmarkConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            batch_size=8,
            max_model_len=128,
            gpu_memory_utilization=0.05,
            additional_config={
                "enable_const_eval": True,
                "cpu_sampling": False,
                "experimental_weight_dtype": "bfp_bf8",
                "optimization_level": 1,
            },
        ),
        id="llama-3.1-8b-instruct-batch8",
    ),
    pytest.param(
        VLLMBenchmarkConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            batch_size=16,
            max_model_len=128,
            gpu_memory_utilization=0.05,
            additional_config={
                "enable_const_eval": True,
                "cpu_sampling": False,
                "experimental_weight_dtype": "bfp_bf8",
                "optimization_level": 1,
            },
        ),
        id="llama-3.1-8b-instruct-batch16",
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


# Sampling comparison: greedy vs non-greedy, device vs CPU, batch=1 and batch=32
# Non-greedy uses temperature=0.6 + repetition_penalty=1.1 (matching demo defaults)
_8B_BASE = dict(model="meta-llama/Llama-3.1-8B-Instruct", max_model_len=128, gpu_memory_utilization=0.05)
_8B_OPT = dict(enable_const_eval=True, experimental_weight_dtype="bfp_bf8", optimization_level=1)

SAMPLING_COMPARISON_CONFIGS = [
    # Batch=1
    pytest.param(VLLMBenchmarkConfig(**_8B_BASE, batch_size=1, additional_config={**_8B_OPT, "cpu_sampling": False}), id="8b-b1-greedy-device"),
    pytest.param(VLLMBenchmarkConfig(**_8B_BASE, batch_size=1, additional_config={**_8B_OPT, "cpu_sampling": True}), id="8b-b1-greedy-cpu"),
    pytest.param(VLLMBenchmarkConfig(**_8B_BASE, batch_size=1, temperature=0.6, repetition_penalty=1.1, additional_config={**_8B_OPT, "cpu_sampling": False}), id="8b-b1-nongreedy-device"),
    pytest.param(VLLMBenchmarkConfig(**_8B_BASE, batch_size=1, temperature=0.6, repetition_penalty=1.1, additional_config={**_8B_OPT, "cpu_sampling": True}), id="8b-b1-nongreedy-cpu"),
    # Batch=32
    pytest.param(VLLMBenchmarkConfig(**_8B_BASE, batch_size=32, additional_config={**_8B_OPT, "cpu_sampling": False}), id="8b-b32-greedy-device"),
    pytest.param(VLLMBenchmarkConfig(**_8B_BASE, batch_size=32, additional_config={**_8B_OPT, "cpu_sampling": True}), id="8b-b32-greedy-cpu"),
    pytest.param(VLLMBenchmarkConfig(**_8B_BASE, batch_size=32, temperature=0.6, repetition_penalty=1.1, additional_config={**_8B_OPT, "cpu_sampling": False}), id="8b-b32-nongreedy-device"),
    pytest.param(VLLMBenchmarkConfig(**_8B_BASE, batch_size=32, temperature=0.6, repetition_penalty=1.1, additional_config={**_8B_OPT, "cpu_sampling": True}), id="8b-b32-nongreedy-cpu"),
]


@pytest.mark.parametrize("config", SAMPLING_COMPARISON_CONFIGS)
def test_sampling_comparison(config, output_file, request):
        _run_vllm_benchmark(config, output_file, request)


# Sampling quality tests: greedy and non-greedy (temp=1.0) device sampling
# for OPT-125M, Llama-3.2-1B, Llama-3.1-8B.  These exist to catch garbage
# output regressions — inspect the printed prompt → output lines to verify.
_OPT_BASE = dict(model="facebook/opt-125m", max_model_len=128, gpu_memory_utilization=0.05)
_1B_BASE = dict(model="meta-llama/Llama-3.2-1B-Instruct", max_model_len=128, gpu_memory_utilization=0.05)
_8B_BASE = dict(model="meta-llama/Llama-3.1-8B-Instruct", max_model_len=128, gpu_memory_utilization=0.05)

_QUALITY_OPTS = dict(
    enable_const_eval=True,
    cpu_sampling=False,
    optimization_level=1,
    experimental_weight_dtype="bfp_bf8",
    # enable_trace=False,
)

SAMPLING_QUALITY_CONFIGS = [
    pytest.param(
        VLLMBenchmarkConfig(**_OPT_BASE, additional_config=_QUALITY_OPTS),
        id="opt125m-greedy-device",
    ),
    pytest.param(
        VLLMBenchmarkConfig(**_OPT_BASE, temperature=1.0, additional_config=_QUALITY_OPTS),
        id="opt125m-nongreedy-device",
    ),
    pytest.param(
        VLLMBenchmarkConfig(**_1B_BASE, additional_config=_QUALITY_OPTS),
        id="llama3.2-1b-greedy-device",
    ),
    pytest.param(
        VLLMBenchmarkConfig(**_1B_BASE, temperature=1.0, additional_config=_QUALITY_OPTS),
        id="llama3.2-1b-nongreedy-device",
    ),
    pytest.param(
        VLLMBenchmarkConfig(**_8B_BASE, additional_config=_QUALITY_OPTS),
        id="llama3.1-8b-greedy-device",
    ),
    pytest.param(
        VLLMBenchmarkConfig(**_8B_BASE, temperature=1.0, additional_config=_QUALITY_OPTS),
        id="llama3.1-8b-nongreedy-device",
    ),
]


@pytest.mark.parametrize("config", SAMPLING_QUALITY_CONFIGS)
def test_sampling_quality(config, output_file, request):
    _run_vllm_benchmark(config, output_file, request)
