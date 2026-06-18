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
from utils import resolve_display_name, sanitize_model_name

# Sampling overrides — keep SINGLE_DEVICE_CONFIGS focused on (model,
# batch_size). CI re-runs the same matrix with different sampling
# configs by setting these env vars (one knob per re-run):
#   TT_BENCHMARK_TEMPERATURE=<float>      default 0.0 (greedy)
#   TT_BENCHMARK_CPU_SAMPLING=1           default 0 (device sampling)
#   TT_BENCHMARK_MAX_MODEL_LEN=<int>      default 128
#   TT_BENCHMARK_KV_CACHE_DTYPE=<str>     default "" (e.g. bfp_bf8, bfp_bf4)
#   _BENCH_OPTIMIZATION_LEVEL=<int>       default 0 (overrides per-test opt level)
#   TT_BENCHMARK_WEIGHT_DTYPE=<str>       e.g. "bfp_bf8"/"bfp_bf4"/"" (overrides per-test weight dtype)
#   TT_BENCHMARK_WEIGHT_OVERRIDES=<path>  JSON file of {glob: dtype} per-tensor mixed-precision overrides
#   TT_BENCHMARK_GMU=<float>              overrides per-test gpu_memory_utilization
#   TT_BENCHMARK_BATCH_SIZE=<int>         overrides per-test batch_size
#   TT_BENCHMARK_TRACE=0|1                overrides per-test enable_trace
_BENCH_TEMPERATURE = float(os.environ.get("TT_BENCHMARK_TEMPERATURE", "0.0"))
_BENCH_CPU_SAMPLING = os.environ.get("TT_BENCHMARK_CPU_SAMPLING", "0") == "1"
_BENCH_MAX_MODEL_LEN = int(os.environ.get("TT_BENCHMARK_MAX_MODEL_LEN", "128"))
_BENCH_MAX_NUM_BATCHED_TOKENS = os.environ.get("TT_BENCHMARK_MAX_NUM_BATCHED_TOKENS")
# Opt in to chunked prefill (tt-xla #4986): set the chunk size. Unset = off.
_BENCH_PREFILL_CHUNK_SIZE = os.environ.get("TT_BENCHMARK_PREFILL_CHUNK_SIZE")
_BENCH_KV_CACHE_DTYPE = os.environ.get("TT_BENCHMARK_KV_CACHE_DTYPE", "")
_BENCH_OPTIMIZATION_LEVEL = os.environ.get("_BENCH_OPTIMIZATION_LEVEL")
_BENCH_WEIGHT_DTYPE = os.environ.get("TT_BENCHMARK_WEIGHT_DTYPE")
_BENCH_WEIGHT_OVERRIDES = os.environ.get("TT_BENCHMARK_WEIGHT_OVERRIDES")
_BENCH_GMU = os.environ.get("TT_BENCHMARK_GMU")
_BENCH_BATCH_SIZE = os.environ.get("TT_BENCHMARK_BATCH_SIZE")
_BENCH_TRACE = os.environ.get("TT_BENCHMARK_TRACE")
_BENCH_FP32_DEST_ACC = os.environ.get("TT_BENCHMARK_FP32_DEST_ACC")


def _config(
    model: str,
    batch_size: int = 32,
    *,
    gpu_memory_utilization: float = 0.05,
    optimization_level: int = 0,
    experimental_weight_dtype: str = "bfp_bf8",
    fp32_dest_acc_en: bool | None = False,
    max_model_len: int | None = None,
    **additional_config_extra,
):
    if _BENCH_OPTIMIZATION_LEVEL is not None:
        optimization_level = int(_BENCH_OPTIMIZATION_LEVEL)
    if _BENCH_BATCH_SIZE is not None:
        batch_size = int(_BENCH_BATCH_SIZE)
    if _BENCH_GMU is not None:
        gpu_memory_utilization = float(_BENCH_GMU)
    additional = {"enable_trace": True}
    if experimental_weight_dtype:
        additional["experimental_weight_dtype"] = experimental_weight_dtype
    if fp32_dest_acc_en is not None:
        additional["fp32_dest_acc_en"] = fp32_dest_acc_en
    if optimization_level > 0:
        additional["optimization_level"] = optimization_level
    # Device sampling (cpu_sampling=False) now works with enable_trace + opt>=1
    # (the tt-xla #4570 guard was removed); opt in to host-side sampling per-run.
    if _BENCH_CPU_SAMPLING:
        additional["cpu_sampling"] = True
    if _BENCH_KV_CACHE_DTYPE:
        additional["experimental_kv_cache_dtype"] = _BENCH_KV_CACHE_DTYPE
    additional.update(additional_config_extra)
    if _BENCH_WEIGHT_DTYPE is not None:
        additional["experimental_weight_dtype"] = _BENCH_WEIGHT_DTYPE
    if _BENCH_WEIGHT_OVERRIDES is not None:
        # Path to a JSON {glob: dtype} file; loaded plugin-side by
        # apply_weight_dtype_overrides. Takes precedence over the uniform dtype.
        additional["weight_dtype_overrides"] = _BENCH_WEIGHT_OVERRIDES
    if _BENCH_TRACE is not None:
        additional["enable_trace"] = _BENCH_TRACE == "1"
    if _BENCH_FP32_DEST_ACC is not None:
        additional["fp32_dest_acc_en"] = _BENCH_FP32_DEST_ACC == "1"
    if _BENCH_PREFILL_CHUNK_SIZE is not None:
        additional["prefill_chunk_size"] = int(_BENCH_PREFILL_CHUNK_SIZE)
    return VLLMBenchmarkConfig(
        model=model,
        batch_size=batch_size,
        max_model_len=(
            max_model_len if max_model_len is not None else _BENCH_MAX_MODEL_LEN
        ),
        max_num_batched_tokens=(
            int(_BENCH_MAX_NUM_BATCHED_TOKENS)
            if _BENCH_MAX_NUM_BATCHED_TOKENS is not None
            else None
        ),
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
    # Allow callers to override weight dtype without passing the same keyword
    # twice to _config (once explicitly and once via **tp_defaults).
    experimental_weight_dtype = tp_defaults.pop("experimental_weight_dtype", "")
    fp32_dest_acc_en = tp_defaults.pop("fp32_dest_acc_en", None)
    return _config(
        model,
        batch_size,
        gpu_memory_utilization=gpu_memory_utilization,
        # Keep TP configs as-is: the single-device alignment defaults
        # (bfp_bf8, fp32_dest_acc_en=False) do not apply here.
        experimental_weight_dtype=experimental_weight_dtype,
        fp32_dest_acc_en=fp32_dest_acc_en,
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
    # Llama
    pytest.param(_config("meta-llama/Llama-3.2-1B-Instruct"), id="llama-3.2-1b"),
    pytest.param(_config("meta-llama/Llama-3.2-3B-Instruct"), id="llama-3.2-3b"),
    pytest.param(_config("meta-llama/Llama-3.1-8B-Instruct"), id="llama-3.1-8b"),
    # Qwen 2.5
    pytest.param(_config("Qwen/Qwen2.5-0.5B-Instruct"), id="qwen2.5-0.5b-instruct"),
    pytest.param(_config("Qwen/Qwen2.5-1.5B-Instruct"), id="qwen2.5-1.5b-instruct"),
    pytest.param(_config("Qwen/Qwen2.5-3B-Instruct"), id="qwen2.5-3b-instruct"),
    pytest.param(_config("Qwen/Qwen2.5-7B-Instruct"), id="qwen2.5-7b-instruct"),
    # Qwen 3
    pytest.param(_config("Qwen/Qwen3-0.6B"), id="qwen3-0.6b"),
    pytest.param(_config("Qwen/Qwen3-1.7B"), id="qwen3-1.7b"),
    pytest.param(_config("Qwen/Qwen3-4B"), id="qwen3-4b"),
    pytest.param(_config("Qwen/Qwen3-8B"), id="qwen3-8b"),
    # Gemma
    pytest.param(_config("google/gemma-1.1-2b-it"), id="gemma-1.1-2b-it"),
    # Phi
    pytest.param(_config("microsoft/phi-1", gpu_memory_utilization=0.30), id="phi-1"),
    pytest.param(
        _config("microsoft/phi-1_5", gpu_memory_utilization=0.30), id="phi-1_5"
    ),
    pytest.param(_config("microsoft/phi-2", gpu_memory_utilization=0.30), id="phi-2"),
    # Falcon 3
    pytest.param(_config("tiiuae/Falcon3-1B-Base"), id="falcon3-1b-base"),
    pytest.param(_config("tiiuae/Falcon3-3B-Base"), id="falcon3-3b-base"),
    pytest.param(_config("tiiuae/Falcon3-7B-Base"), id="falcon3-7b-base"),
    # Mistral
    pytest.param(
        _config("mistralai/Mistral-7B-Instruct-v0.3"), id="mistral-7b-instruct"
    ),
    pytest.param(_config("mistralai/Ministral-8B-Instruct-2410"), id="ministral-8b"),
    # OPT (vLLM-only fast canary; not part of the torch-xla matrix)
    pytest.param(_config("facebook/opt-125m"), id="opt-125m"),
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
        _tp_config("Qwen/Qwen3-32B", 1, use_2d_mesh=False), id="qwen3-32b-qb2-tp"
    ),
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
    pytest.param(
        _tp_config("meta-llama/Llama-3.1-8B-Instruct", 1),
        id="llama-3.1-8b-tp",
    ),
    pytest.param(
        _tp_config(
            "meta-llama/Llama-3.1-70B-Instruct",
            1,
            enable_const_eval=True,
            experimental_weight_dtype="bfp_bf8",
        ),
        id="llama-3.1-70b-tp",
    ),
    # Verify fused decode_postprocess compiles to expected graph count (cpu_sampling=False path)
    pytest.param(
        _config("facebook/opt-125m", 1, gpu_memory_utilization=0.001),
        id="opt-125m-fused-measure",
    ),
]


def _run_vllm_benchmark(config, output_file, request):
    resolved_display_name = resolve_display_name(request=request, fallback=config.model)
    display_name = (
        resolved_display_name
        if resolved_display_name.startswith("vllm_")
        else f"vllm_{resolved_display_name}"
    )

    print(f"\n{'='*60}")
    print(f"vLLM Benchmark: {display_name}")
    print(f"{'='*60}")

    # Dump compiler IR modules.
    config.additional_config.setdefault("export_path", "modules")
    config.additional_config.setdefault(
        "export_model_name", sanitize_model_name(display_name)
    )

    results = benchmark_vllm(config, display_name)

    if output_file:
        results["project"] = "tt-forge/tt-xla"
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
    resolved_display_name = resolve_display_name(request=request, fallback=config.model)
    display_name = (
        resolved_display_name
        if resolved_display_name.startswith("vllm_")
        else f"vllm_{resolved_display_name}"
    )
    results = benchmark_vllm_embedding(config, display_name)
    if output_file:
        results["project"] = "tt-forge/tt-xla"
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


# ---------------------------------------------------------------------------
# Qwen3-8B decode-rate sweep (single chip P150). Mirrors the additional_config
# from tt-inference-server's launch_qwen3_8b.sh so this is a faithful tt-xla
# repro of the online server's decode path, minus the serving/streaming layer:
#   opt=1, gmu=0.30, weights bfp8, KV bfp8, on-device chunked prefill 2048,
#   fp32_dest_acc_en=false, enable_trace, device sampling, enable_const_eval.
#
# num_hidden_layers=1 trims the model to a single decoder layer (NUM_HIDDEN_LAYERS=1
# on the server) so device matmul is negligible. That isolates any per-decode-step
# / host overhead that grows with the number of GENERATED tokens — the suspected
# cause of the decode-rate decay seen on the online server (decode tok/s falls as
# OSL grows, yet is flat vs prefilled ISL depth).
#
# Sweep OSL at fixed ISL and read avg_tokens_per_sec ("samples_per_sec" in the
# result) — decode tok/s the engine computes from last_token_ts - first_token_ts,
# i.e. excluding prefill. If it falls as OSL grows, the slowdown reproduces in
# pure tt-xla (engine/device), not just the serving layer. Each (isl, osl) is an
# individual test.
# (isl, osl) datapoints — ISL fixed, OSL swept.
_QWEN3_8B_DECODE_PAIRS = [(128, 128), (128, 1024), (128, 4096)]


def _round_up_256(n: int) -> int:
    return ((n + 255) // 256) * 256


# ONE fixed context length for the whole sweep so the compiled graph/config is
# identical across every point and only max_tokens (OSL) varies. Deriving
# max_model_len per-OSL changes the graph AND, once it exceeds prefill_chunk_size
# (2048), tt-xla chunked prefill requires max_model_len % 256 == 0 — e.g. 4736
# raises NotImplementedError. Round up to a multiple of 256 with headroom.
_SWEEP_MAX_MODEL_LEN = _round_up_256(
    max(i + o for i, o in _QWEN3_8B_DECODE_PAIRS) + 256
)


def _qwen3_8b_decode_config(isl: int, osl: int):
    cfg = _config(
        "Qwen/Qwen3-8B",
        batch_size=1,
        gpu_memory_utilization=0.30,
        optimization_level=1,
        experimental_weight_dtype="bfp_bf8",
        fp32_dest_acc_en=False,
        # Fixed across the sweep (multiple of 256) so the graph is identical for
        # every OSL — only max_tokens changes between points.
        max_model_len=_SWEEP_MAX_MODEL_LEN,
        experimental_kv_cache_dtype="bfp_bf8",
        prefill_chunk_size=2048,
        enable_const_eval=True,
        min_context_len=32,
        num_hidden_layers=1,
    )
    cfg.max_tokens = osl
    # "word " is ~1 BPE token, so this is ~isl input tokens. ISL barely affects
    # decode rate (decode is flat vs prefilled depth), so approximate ISL is fine;
    # the measured prompt_tokens is printed by the benchmark.
    cfg.prompt = ("word " * isl).strip()
    return cfg


_QWEN3_8B_DECODE_SWEEP = [
    pytest.param(isl, osl, id=f"isl{isl}_osl{osl}")
    for isl, osl in _QWEN3_8B_DECODE_PAIRS
]


@pytest.mark.parametrize("isl,osl", _QWEN3_8B_DECODE_SWEEP)
def test_vllm_qwen3_8b_decode_sweep(isl, osl, output_file, request):
    _run_vllm_benchmark(_qwen3_8b_decode_config(isl, osl), output_file, request)


# Same OSL sweep, but with the engine configured like the tt-inference-server
# online server (max_num_seqs=32, max_model_len=40960, min_context_len=128) while
# submitting a SINGLE request (num_prompts=1). IR comparison showed the online
# decode graph runs at batch 32 (32x4096 hidden, 32x151936 logits/sampling every
# token) vs batch 1 offline — this variant reproduces that batch-32 / large-context
# graph. Purpose: if decode tok/s slows with OSL here, the slowdown is the engine
# config (batch-32 / big context); if it stays flat (like the batch-1 sweep), the
# slowdown is in the serving path, not tt-xla.
def _qwen3_8b_decode_config_match_online(isl: int, osl: int):
    cfg = _config(
        "Qwen/Qwen3-8B",
        batch_size=32,  # -> max_num_seqs=32, like the server's MAX_NUM_SEQS
        gpu_memory_utilization=0.30,
        optimization_level=1,
        experimental_weight_dtype="bfp_bf8",
        fp32_dest_acc_en=False,
        max_model_len=40960,  # server MAX_MODEL_LENGTH (multiple of 256)
        experimental_kv_cache_dtype="bfp_bf8",
        prefill_chunk_size=2048,
        enable_const_eval=True,
        min_context_len=128,  # server min_context_len
        num_hidden_layers=1,
    )
    cfg.num_prompts = 1  # ONE active request on the batch-32 engine
    cfg.max_num_batched_tokens = 2048  # chunked-prefill cap, like the server
    cfg.max_tokens = osl
    cfg.prompt = ("word " * isl).strip()
    return cfg


@pytest.mark.parametrize("isl,osl", _QWEN3_8B_DECODE_SWEEP)
def test_vllm_qwen3_8b_decode_sweep_match_online(isl, osl, output_file, request):
    _run_vllm_benchmark(
        _qwen3_8b_decode_config_match_online(isl, osl), output_file, request
    )
