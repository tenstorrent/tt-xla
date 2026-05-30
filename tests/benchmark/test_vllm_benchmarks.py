# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import os

import pytest
from benchmarks.vllm_benchmark import (
    VLLMBenchmarkConfig,
    VLLMEmbeddingBenchmarkConfig,
    benchmark_vllm,
    benchmark_vllm_embedding,
)
from utils import resolve_display_name

# Sampling overrides — keep SINGLE_DEVICE_CONFIGS focused on (model,
# batch_size). CI re-runs the same matrix with different sampling
# configs by setting these env vars (one knob per re-run):
#   TT_BENCHMARK_TEMPERATURE=<float>      default 0.0 (greedy)
#   TT_BENCHMARK_CPU_SAMPLING=1           default 0 (device sampling)
#   TT_BENCHMARK_MAX_MODEL_LEN=<int>      default 128
#   _BENCH_OPTIMIZATION_LEVEL=<int>       default 0 (overrides per-test opt level)
_BENCH_TEMPERATURE = float(os.environ.get("TT_BENCHMARK_TEMPERATURE", "0.0"))
_BENCH_CPU_SAMPLING = os.environ.get("TT_BENCHMARK_CPU_SAMPLING", "0") == "1"
_BENCH_MAX_MODEL_LEN = int(os.environ.get("TT_BENCHMARK_MAX_MODEL_LEN", "128"))
_BENCH_OPTIMIZATION_LEVEL = os.environ.get("_BENCH_OPTIMIZATION_LEVEL")


def _config(
    model: str,
    batch_size: int,
    *,
    gpu_memory_utilization: float = 0.05,
    optimization_level: int = 0,
    **additional_config_extra,
):
    if _BENCH_OPTIMIZATION_LEVEL is not None:
        optimization_level = int(_BENCH_OPTIMIZATION_LEVEL)
    additional = {"enable_trace": True}
    if optimization_level > 0:
        additional["optimization_level"] = optimization_level
        # TTConfig raises if enable_trace=True AND opt>=1 AND cpu_sampling=False
        additional["cpu_sampling"] = True
    if _BENCH_CPU_SAMPLING:
        additional["cpu_sampling"] = True
    additional.update(additional_config_extra)
    return VLLMBenchmarkConfig(
        model=model,
        batch_size=batch_size,
        max_model_len=_BENCH_MAX_MODEL_LEN,
        gpu_memory_utilization=gpu_memory_utilization,
        temperature=_BENCH_TEMPERATURE,
        additional_config=additional,
    )


def _tp_config(
    model: str,
    batch_size: int,
    *,
    gpu_memory_utilization: float = 0.005,
    **additional_config_extra,
):
    tp_defaults = {
        "enable_tensor_parallel": True,
        "use_2d_mesh": True,
        "min_context_len": 32,
    }
    tp_defaults.update(additional_config_extra)
    return _config(
        model,
        batch_size,
        gpu_memory_utilization=gpu_memory_utilization,
        **tp_defaults,
    )


def _gemma4_tp_config(model: str, batch_size: int):
    # Gemma-4 is a multimodal model run text-only on a TP mesh. Mirrors
    # tests/integrations/vllm_plugin/generative/test_tensor_parallel_generation.py
    # ::test_tensor_parallel_generation_bhqb_gemma4_31b:
    #   - limit_mm_per_prompt zeroed so the vision/audio tower never compiles
    #   - max_num_batched_tokens floored at 2560 (MultiModalBudget video floor)
    #   - flat_model_io for Gemma-4's PLE forward; use_2d_mesh=False -> 1D mesh
    cfg = _config(
        model,
        batch_size,
        gpu_memory_utilization=0.1,
        enable_tensor_parallel=True,
        use_2d_mesh=False,
        pad_attention_heads=True,
        # Gemma-4 has two distinct attention configurations (sliding /
        # full+k_eq_v). The min-cost padding strategy produces unequal-sized
        # Q vs K/V, which trips a tt-metal concat kernel placement bug on
        # the [Q;K;V] concat in XlaQKVParallelLinear. Force equal-sized
        # padding (c=k) to stay on a working kernel path.
        pad_attention_heads_force_equal=True,
        min_context_len=32,
        enable_const_eval=True,
        experimental_weight_dtype="",
        cpu_sampling=False,
        flat_model_io=True,
    )
    cfg.limit_mm_per_prompt = {"image": 0, "video": 0, "audio": 0}
    cfg.min_num_batched_tokens = 2560
    # Gemma-4-it is instruct-tuned; drive via the chat template so it
    # produces coherent output instead of a degenerate completion loop.
    cfg.use_chat_template = True
    return cfg


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
    pytest.param(
        _config(
            "facebook/opt-125m", 1, gpu_memory_utilization=0.001, optimization_level=1
        ),
        id="opt-125m-opt1",
    ),
    pytest.param(
        _config(
            "facebook/opt-125m", 32, gpu_memory_utilization=0.02, optimization_level=1
        ),
        id="opt-125m-batch32-opt1",
        marks=pytest.mark.xfail(
            reason="tt-mlir MemoryLayoutPropagation::consolidateBeam assert "
            "(regression from tt-mlir uplift #4569); see tt-mlir issue TODO",
            strict=False,
            run=True,
        ),
    ),
    pytest.param(_config("Qwen/Qwen2.5-0.5B-Instruct", 1), id="qwen2.5-0.5b-instruct"),
    pytest.param(_config("Qwen/Qwen2.5-1.5B-Instruct", 1), id="qwen2.5-1.5b-instruct"),
    pytest.param(_config("Qwen/Qwen2.5-3B-Instruct", 1), id="qwen2.5-3b-instruct"),
    pytest.param(_config("Qwen/Qwen3-0.6B", 1), id="qwen3-0.6b"),
    pytest.param(_config("Qwen/Qwen3-1.7B", 1), id="qwen3-1.7b"),
    pytest.param(_config("microsoft/phi-1", 1), id="phi-1"),
    pytest.param(_config("microsoft/phi-1_5", 1), id="phi-1_5"),
    pytest.param(_config("microsoft/phi-2", 1), id="phi-2"),
    pytest.param(_config("tiiuae/Falcon3-1B-Base", 1), id="falcon3-1b-base"),
    pytest.param(
        _config("tiiuae/Falcon3-1B-Base", 1, optimization_level=1),
        id="falcon3-1b-base-opt1",
    ),
    pytest.param(_config("tiiuae/Falcon3-3B-Base", 1), id="falcon3-3b-base"),
]


TP_CONFIGS = [
    pytest.param(_tp_config("tiiuae/Falcon3-7B-Base", 1), id="falcon3-7b-tp"),
    pytest.param(_tp_config("tiiuae/Falcon3-10B-Base", 1), id="falcon3-10b-tp"),
    pytest.param(_tp_config("Qwen/Qwen3-8B", 1), id="qwen3-8b-tp"),
    pytest.param(
        _tp_config("Qwen/Qwen3-8B", 1, optimization_level=1),
        id="qwen3-8b-tp-opt1",
    ),
    pytest.param(_tp_config("Qwen/Qwen3-14B", 1), id="qwen3-14b-tp"),
    pytest.param(_tp_config("Qwen/Qwen3-32B", 1), id="qwen3-32b-tp"),
    pytest.param(_gemma4_tp_config("google/gemma-4-31B-it", 1), id="gemma4-31b-it-tp"),
    pytest.param(
        _tp_config("Qwen/Qwen2.5-14B-Instruct", 1), id="qwen2.5-14b-instruct-tp"
    ),
    pytest.param(
        _tp_config("Qwen/Qwen2.5-Coder-32B-Instruct", 1),
        id="qwen2.5-coder-32b-instruct-tp",
    ),
    pytest.param(
        _tp_config("mistralai/Ministral-8B-Instruct-2410", 1), id="ministral-8b-tp"
    ),
    pytest.param(
        _tp_config("mistralai/Mistral-Nemo-Instruct-2407", 1),
        id="mistral-nemo-instruct-2407-tp",
    ),
    pytest.param(
        _tp_config("mistralai/Mistral-Small-24B-Instruct-2501", 1),
        id="mistral-small-24b-instruct-2501-tp",
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


def _embedding_config(
    model: str,
    batch_size: int,
    *,
    max_model_len: int = 512,
    gpu_memory_utilization: float = 0.05,
    **additional_config_extra,
):
    additional = {"enable_trace": True}
    additional.update(additional_config_extra)
    return VLLMEmbeddingBenchmarkConfig(
        model=model,
        batch_size=batch_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        additional_config=additional,
    )


def _run_vllm_embedding_benchmark(config, output_file, request):
    display_name = "vllm_" + resolve_display_name(
        request=request, fallback=config.model
    )
    results = benchmark_vllm_embedding(config, display_name)
    if output_file:
        results["project"] = "tt-xla"
        results["model_rawname"] = config.model
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to {output_file}")


# Trace disabled: host/device tensor shape mismatch (https://github.com/tenstorrent/tt-xla/issues/3936)
def test_vllm_qwen3_embedding_4b_batch1(output_file, request):
    _run_vllm_embedding_benchmark(
        _embedding_config(
            "Qwen/Qwen3-Embedding-4B", 1, max_model_len=128, enable_trace=False
        ),
        output_file,
        request,
    )


def test_vllm_bge_m3_batch1(output_file, request):
    _run_vllm_embedding_benchmark(
        _embedding_config("BAAI/bge-m3", 1),
        output_file,
        request,
    )


def test_vllm_bge_m3_batch32(output_file, request):
    _run_vllm_embedding_benchmark(
        _embedding_config("BAAI/bge-m3", 32),
        output_file,
        request,
    )


@pytest.mark.parametrize("config", SINGLE_DEVICE_CONFIGS)
def test_vllm_benchmark(config, output_file, request):
    _run_vllm_benchmark(config, output_file, request)


@pytest.mark.parametrize("config", TP_CONFIGS)
def test_vllm_tp_benchmark(config, output_file, request):
    _run_vllm_benchmark(config, output_file, request)
