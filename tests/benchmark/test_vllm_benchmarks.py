# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses
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


def _run_vllm_benchmark(config, output_file, request, max_output_tokens=None):
    if max_output_tokens is not None:
        config = dataclasses.replace(
            config, max_tokens=max_output_tokens, warmup_iterations=0
        )

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
def test_vllm_benchmark(config, output_file, request, max_output_tokens):
    _run_vllm_benchmark(config, output_file, request, max_output_tokens)


# Sampling comparison: greedy vs non-greedy, device vs CPU, batch=1 and batch=32
# Non-greedy uses temperature=0.6 + repetition_penalty=1.1 (matching demo defaults)
_8B_BASE = dict(
    model="meta-llama/Llama-3.1-8B-Instruct",
    max_model_len=128,
    gpu_memory_utilization=0.05,
)
_8B_OPT = dict(
    enable_const_eval=True, experimental_weight_dtype="bfp_bf8", optimization_level=1
)

SAMPLING_COMPARISON_CONFIGS = [
    # Batch=1
    pytest.param(
        VLLMBenchmarkConfig(
            **_8B_BASE,
            batch_size=1,
            additional_config={**_8B_OPT, "cpu_sampling": False},
        ),
        id="8b-b1-greedy-device",
    ),
    pytest.param(
        VLLMBenchmarkConfig(
            **_8B_BASE,
            batch_size=1,
            additional_config={**_8B_OPT, "cpu_sampling": True},
        ),
        id="8b-b1-greedy-cpu",
    ),
    pytest.param(
        VLLMBenchmarkConfig(
            **_8B_BASE,
            batch_size=1,
            temperature=0.6,
            repetition_penalty=1.1,
            additional_config={**_8B_OPT, "cpu_sampling": False},
        ),
        id="8b-b1-nongreedy-device",
    ),
    pytest.param(
        VLLMBenchmarkConfig(
            **_8B_BASE,
            batch_size=1,
            temperature=0.6,
            additional_config={**_8B_OPT, "cpu_sampling": False},
        ),
        id="8b-b1-nongreedy-nopenalty-device",
    ),
    pytest.param(
        VLLMBenchmarkConfig(
            **_8B_BASE,
            batch_size=1,
            temperature=0.6,
            repetition_penalty=1.1,
            additional_config={**_8B_OPT, "cpu_sampling": True},
        ),
        id="8b-b1-nongreedy-cpu",
    ),
    # Batch=32
    pytest.param(
        VLLMBenchmarkConfig(
            **_8B_BASE,
            batch_size=32,
            additional_config={**_8B_OPT, "cpu_sampling": False},
        ),
        id="8b-b32-greedy-device",
    ),
    pytest.param(
        VLLMBenchmarkConfig(
            **_8B_BASE,
            batch_size=32,
            additional_config={**_8B_OPT, "cpu_sampling": True},
        ),
        id="8b-b32-greedy-cpu",
    ),
    pytest.param(
        VLLMBenchmarkConfig(
            **_8B_BASE,
            batch_size=32,
            temperature=0.6,
            repetition_penalty=1.1,
            additional_config={**_8B_OPT, "cpu_sampling": False},
        ),
        id="8b-b32-nongreedy-device",
    ),
    pytest.param(
        VLLMBenchmarkConfig(
            **_8B_BASE,
            batch_size=32,
            temperature=0.6,
            repetition_penalty=1.1,
            additional_config={**_8B_OPT, "cpu_sampling": True},
        ),
        id="8b-b32-nongreedy-cpu",
    ),
]


@pytest.mark.parametrize("config", SAMPLING_COMPARISON_CONFIGS)
def test_sampling_comparison(config, output_file, request, max_output_tokens):
    _run_vllm_benchmark(config, output_file, request, max_output_tokens)


# Same configs with trace enabled
_8B_OPT_TRACE = {**_8B_OPT, "enable_trace": True}

SAMPLING_COMPARISON_TRACE_CONFIGS = [
    pytest.param(
        VLLMBenchmarkConfig(
            **_8B_BASE,
            batch_size=1,
            temperature=0.6,
            repetition_penalty=1.1,
            additional_config={**_8B_OPT_TRACE, "cpu_sampling": False},
        ),
        id="8b-b1-nongreedy-device-trace",
    ),
    pytest.param(
        VLLMBenchmarkConfig(
            **_8B_BASE,
            batch_size=32,
            temperature=0.6,
            repetition_penalty=1.1,
            additional_config={**_8B_OPT_TRACE, "cpu_sampling": False},
        ),
        id="8b-b32-nongreedy-device-trace",
    ),
]


@pytest.mark.parametrize("config", SAMPLING_COMPARISON_TRACE_CONFIGS)
def test_sampling_comparison_trace(config, output_file, request, max_output_tokens):
    _run_vllm_benchmark(config, output_file, request, max_output_tokens)


# OPT-125M: fast pipecleaning model for sampling integration
_OPT_BASE = dict(
    model="facebook/opt-125m", max_model_len=128, gpu_memory_utilization=0.05
)
_OPT_OPT = dict(enable_const_eval=True, optimization_level=1)

OPT_SAMPLING_CONFIGS = [
    pytest.param(
        VLLMBenchmarkConfig(
            **_OPT_BASE,
            batch_size=1,
            additional_config={**_OPT_OPT, "cpu_sampling": False},
        ),
        id="opt125m-b1-greedy-device",
    ),
    pytest.param(
        VLLMBenchmarkConfig(
            **_OPT_BASE,
            batch_size=1,
            temperature=0.6,
            repetition_penalty=1.1,
            additional_config={**_OPT_OPT, "cpu_sampling": False},
        ),
        id="opt125m-b1-nongreedy-device",
    ),
    pytest.param(
        VLLMBenchmarkConfig(
            **_OPT_BASE,
            batch_size=1,
            temperature=0.6,
            repetition_penalty=1.1,
            additional_config={**_OPT_OPT, "cpu_sampling": True},
        ),
        id="opt125m-b1-nongreedy-cpu",
    ),
]


@pytest.mark.parametrize("config", OPT_SAMPLING_CONFIGS)
def test_opt_sampling(config, output_file, request, max_output_tokens):
    _run_vllm_benchmark(config, output_file, request, max_output_tokens)
