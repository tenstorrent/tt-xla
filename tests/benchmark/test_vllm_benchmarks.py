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


# Sampling quality tests: greedy and non-greedy (temp=1.0) device sampling
# for OPT-125M, Llama-3.2-1B, Llama-3.1-8B.  These exist to catch garbage
# output regressions — inspect the printed prompt → output lines to verify.
_OPT_BASE = dict(
    model="facebook/opt-125m", max_model_len=128, gpu_memory_utilization=0.05
)
_1B_BASE = dict(
    model="meta-llama/Llama-3.2-1B-Instruct",
    max_model_len=128,
    gpu_memory_utilization=0.05,
)
_3B_BASE = dict(
    model="meta-llama/Llama-3.2-3B-Instruct",
    max_model_len=128,
    gpu_memory_utilization=0.05,
)
_8B_BASE = dict(
    model="meta-llama/Llama-3.1-8B-Instruct",
    max_model_len=128,
    gpu_memory_utilization=0.05,
)

_QUALITY_OPTS = dict(
    enable_const_eval=True,
    cpu_sampling=False,
    optimization_level=1,
    experimental_weight_dtype="bfp_bf8",
    enable_trace=True,
)

_QUALITY_OPTS_CPU = dict(_QUALITY_OPTS, cpu_sampling=True)

# (variant_suffix, additional_config, extra_kwargs)
_QUALITY_VARIANTS = [
    ("greedy-device", _QUALITY_OPTS, {}),
    ("nongreedy-device", _QUALITY_OPTS, {"temperature": 1.0}),
    ("greedy-cpu", _QUALITY_OPTS_CPU, {}),
    ("nongreedy-cpu", _QUALITY_OPTS_CPU, {"temperature": 1.0}),
]


def _quality_params(label, base, batch_size=None):
    batch_kwargs = {"batch_size": batch_size} if batch_size is not None else {}
    batch_suffix = f"-b{batch_size}" if batch_size is not None else ""
    return [
        pytest.param(
            VLLMBenchmarkConfig(
                **base, **batch_kwargs, **extra, additional_config=opts
            ),
            id=f"{label}{batch_suffix}-{name}",
        )
        for name, opts, extra in _QUALITY_VARIANTS
    ]


SAMPLING_QUALITY_CONFIGS = [
    *_quality_params("opt125m", _OPT_BASE),
    *_quality_params("opt125m", _OPT_BASE, batch_size=2),
    *_quality_params("opt125m", _OPT_BASE, batch_size=32),
    *_quality_params("llama3.2-1b", _1B_BASE),
    *_quality_params("llama3.2-1b", _1B_BASE, batch_size=2),
    *_quality_params("llama3.2-1b", _1B_BASE, batch_size=32),
    *_quality_params("llama3.2-3b", _3B_BASE),
    *_quality_params("llama3.2-3b", _3B_BASE, batch_size=2),
    *_quality_params("llama3.2-3b", _3B_BASE, batch_size=32),
    *_quality_params("llama3.1-8b", _8B_BASE),
    *_quality_params("llama3.1-8b", _8B_BASE, batch_size=2),
    *_quality_params("llama3.1-8b", _8B_BASE, batch_size=32),
]


@pytest.mark.parametrize("config", SAMPLING_QUALITY_CONFIGS)
def test_sampling_quality(config, output_file, request):
    _run_vllm_benchmark(config, output_file, request)
