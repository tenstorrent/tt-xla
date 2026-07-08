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
#   TT_BENCHMARK_KV_CACHE_DTYPE=<str>     e.g. "bfp_bf8"/"bfp_bf4"/"" (overrides per-test kv cache dtype)
#   _BENCH_OPTIMIZATION_LEVEL=<int>       default 0 (overrides per-test opt level)
#   TT_BENCHMARK_WEIGHT_DTYPE=<str>       e.g. "bfp_bf8"/"bfp_bf4"/"" (overrides per-test weight dtype)
#   TT_BENCHMARK_WEIGHT_OVERRIDES=<path>  JSON file of {glob: dtype} per-tensor mixed-precision overrides
#   TT_BENCHMARK_GMU=<float>              overrides per-test gpu_memory_utilization
#   TT_BENCHMARK_BATCH_SIZE=<int>         overrides per-test batch_size
#   TT_BENCHMARK_TRACE=0|1                overrides per-test enable_trace
_BENCH_TEMPERATURE = float(os.environ.get("TT_BENCHMARK_TEMPERATURE", "0.0"))
_BENCH_CPU_SAMPLING = os.environ.get("TT_BENCHMARK_CPU_SAMPLING", "0") == "1"
_BENCH_MAX_MODEL_LEN = int(os.environ.get("TT_BENCHMARK_MAX_MODEL_LEN", "128"))
_BENCH_KV_CACHE_DTYPE = os.environ.get("TT_BENCHMARK_KV_CACHE_DTYPE")
_BENCH_OPTIMIZATION_LEVEL = os.environ.get("_BENCH_OPTIMIZATION_LEVEL")
_BENCH_WEIGHT_DTYPE = os.environ.get("TT_BENCHMARK_WEIGHT_DTYPE")
_BENCH_WEIGHT_OVERRIDES = os.environ.get("TT_BENCHMARK_WEIGHT_OVERRIDES")
_BENCH_GMU = os.environ.get("TT_BENCHMARK_GMU")
_BENCH_BATCH_SIZE = os.environ.get("TT_BENCHMARK_BATCH_SIZE")
_BENCH_TRACE = os.environ.get("TT_BENCHMARK_TRACE")
# Opt-in chunked prefill (tt-xla #4986): caps per-step prefill budget so
# compile time + peak prefill DRAM are bounded by the chunk size, not
# max_model_len. 0 / unset = disabled (unchanged behavior).
_BENCH_PREFILL_CHUNK_SIZE = os.environ.get("TT_BENCHMARK_PREFILL_CHUNK_SIZE")


def _config(
    model: str,
    batch_size: int = 32,
    *,
    gpu_memory_utilization: float = 0.05,
    optimization_level: int = 2,
    experimental_weight_dtype: str = "bfp_bf8",
    experimental_kv_cache_dtype: str = "bfp_bf8",
    # None => leave the compute-kernel-config knob unset so compiler
    # can set the default value or leave it up to ttnn.
    # Only set this deliberately, never as a frontend default.
    fp32_dest_acc_en: bool | None = None,
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
    if experimental_kv_cache_dtype:
        additional["experimental_kv_cache_dtype"] = experimental_kv_cache_dtype
    if fp32_dest_acc_en is not None:
        additional["fp32_dest_acc_en"] = fp32_dest_acc_en
    if optimization_level:
        additional["optimization_level"] = optimization_level
    if _BENCH_CPU_SAMPLING:
        additional["cpu_sampling"] = True
    additional.update(additional_config_extra)
    if _BENCH_WEIGHT_DTYPE is not None:
        additional["experimental_weight_dtype"] = _BENCH_WEIGHT_DTYPE
    if _BENCH_KV_CACHE_DTYPE is not None:
        additional["experimental_kv_cache_dtype"] = _BENCH_KV_CACHE_DTYPE
    if _BENCH_WEIGHT_OVERRIDES is not None:
        # Path to a JSON {glob: dtype} file; loaded plugin-side by
        # apply_weight_dtype_overrides. Takes precedence over the uniform dtype.
        additional["weight_dtype_overrides"] = _BENCH_WEIGHT_OVERRIDES
    if _BENCH_TRACE is not None:
        additional["enable_trace"] = _BENCH_TRACE == "1"
    if _BENCH_PREFILL_CHUNK_SIZE is not None:
        additional["prefill_chunk_size"] = int(_BENCH_PREFILL_CHUNK_SIZE)
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
    gpu_memory_utilization: float = 0.1,
    optimization_level: int = 2,
    **additional_config_extra,
):
    tp_defaults = {
        "enable_tensor_parallel": True,
        "use_2d_mesh": False,
        # Default to a 1D mesh (1, num_devices); each TP config passes an
        # explicit mesh_shape for its target machine.
        "mesh_shape": None,
        "min_context_len": 32,
    }
    tp_defaults.update(additional_config_extra)
    # Allow callers to override weight/kv-cache dtype without passing the same
    # keyword twice to _config (once explicitly and once via **tp_defaults).
    experimental_weight_dtype = tp_defaults.pop("experimental_weight_dtype", "")
    experimental_kv_cache_dtype = tp_defaults.pop("experimental_kv_cache_dtype", "")
    fp32_dest_acc_en = tp_defaults.pop("fp32_dest_acc_en", None)
    return _config(
        model,
        batch_size,
        gpu_memory_utilization=gpu_memory_utilization,
        optimization_level=optimization_level,
        # Keep TP configs as-is: the single-device alignment default
        # (bfp_bf8 weight dtype) does not apply here.
        experimental_weight_dtype=experimental_weight_dtype,
        experimental_kv_cache_dtype=experimental_kv_cache_dtype,
        fp32_dest_acc_en=fp32_dest_acc_en,
        **tp_defaults,
    )


def _gemma4_tp_config(model: str, batch_size: int):
    # Gemma-4 is a multimodal model run text-only on a TP mesh. Mirrors
    # tests/integrations/vllm_plugin/generative/test_tensor_parallel_generation.py
    # ::test_tensor_parallel_generation_gemma4_31b:
    #   - limit_mm_per_prompt zeroed so the vision/audio tower never compiles
    #   - max_num_batched_tokens floored at 2560 (MultiModalBudget video floor)
    #   - flat_model_io for Gemma-4's PLE forward; use_2d_mesh=False -> 1D mesh
    #     (gemma4 TP runs on qb2-blackhole). mesh_shape=None alone is NOT
    #     enough for a 1D mesh -- TTConfig.use_2d_mesh defaults to True, so
    #     without this explicit False, determine_mesh_shape() picks a 2D
    #     (2, 2) mesh for 4 devices instead of the intended 1D (1, 4).
    cfg = _config(
        model,
        batch_size,
        gpu_memory_utilization=0.2,
        # opt-level 2 fails with an L1 out-of-memory TT_FATAL; see #5440.
        optimization_level=1,
        enable_tensor_parallel=True,
        use_2d_mesh=False,
        min_context_len=32,
        enable_const_eval=True,
        experimental_weight_dtype="",
        experimental_kv_cache_dtype="",
        cpu_sampling=False,
        flat_model_io=True,
    )
    cfg.limit_mm_per_prompt = {"image": 0, "video": 0, "audio": 0}
    cfg.min_num_batched_tokens = 2560
    # Gemma-4-it is instruct-tuned; drive via the chat template so it
    # produces coherent output instead of a degenerate completion loop.
    cfg.use_chat_template = True
    return cfg


def _mistral_small_31_tp_config(model: str, batch_size: int):
    # Mistral-Small-3.1 is a Pixtral-based multimodal model, benchmarked
    # text-only (limit_mm_per_prompt zeroed so the vision tower never compiles),
    # mirroring _gemma4_tp_config. Runs on galaxy-wh-6u in a 8x4 mesh.
    #
    # Validated max_model_len of 8192 at GMU of 0.16, but the current default
    # of max_model_len is 128, so it needs to be overriden through the env. var.
    cfg = _config(
        model,
        batch_size,
        gpu_memory_utilization=0.16,
        optimization_level=1,
        experimental_weight_dtype="bfp_bf8",
        experimental_kv_cache_dtype="bfp_bf8",
        enable_tensor_parallel=True,
        use_2d_mesh=False,
        mesh_shape=[8, 4],
        min_context_len=32,
        enable_const_eval=True,
        # b1-prefill optimization: serve prefills serially (small graph) when
        # <=16 are pending instead of a wasted-row b32 batch. Needs min_num_seqs
        # < max_num_seqs (batch_size=32).
        min_num_seqs=1,
        prefill_batch_threshold=16,
    )
    cfg.limit_mm_per_prompt = {"image": 0}
    # Instruct-tuned: drive via the chat template for coherent output.
    cfg.use_chat_template = True
    return cfg


def _qwen3_4b_production_config():
    cfg = _config(
        "Qwen/Qwen3-4B",
        32,
        gpu_memory_utilization=0.5,
        optimization_level=1,
        experimental_weight_dtype="bfp_bf8",
        experimental_kv_cache_dtype="bfp_bf8",
        enable_const_eval=True,
        min_context_len=128,
        prefill_chunk_size=1024,
        min_num_seqs=1,
        prefill_batch_threshold=16,
        max_prefill_num_seqs=16,
    )
    cfg.max_model_len = 40960
    cfg.use_chat_template = True
    return cfg


def _qwen3_8b_production_config():
    cfg = _config(
        "Qwen/Qwen3-8B",
        32,
        gpu_memory_utilization=0.35,
        optimization_level=1,
        experimental_weight_dtype="bfp_bf8",
        experimental_kv_cache_dtype="bfp_bf8",
        enable_const_eval=True,
        min_context_len=128,
        prefill_chunk_size=1024,
        min_num_seqs=1,
        prefill_batch_threshold=16,
        max_prefill_num_seqs=16,
    )
    cfg.max_model_len = 40960
    cfg.use_chat_template = True
    return cfg


def _llama3_2_3b_instruct_production_config():
    cfg = _config(
        "meta-llama/Llama-3.2-3B-Instruct",
        32,
        gpu_memory_utilization=0.5,
        optimization_level=1,
        experimental_weight_dtype="bfp_bf8",
        experimental_kv_cache_dtype="bfp_bf8",
        enable_const_eval=True,
        min_context_len=128,
        prefill_chunk_size=1024,
        min_num_seqs=1,
        prefill_batch_threshold=16,
        max_prefill_num_seqs=16,
    )
    cfg.max_model_len = 65536
    cfg.use_chat_template = True
    return cfg


def _llama3_1_8b_instruct_production_config():
    cfg = _config(
        "meta-llama/Llama-3.1-8B-Instruct",
        32,
        gpu_memory_utilization=0.35,
        optimization_level=1,
        experimental_weight_dtype="bfp_bf8",
        experimental_kv_cache_dtype="bfp_bf8",
        enable_const_eval=True,
        min_context_len=128,
        prefill_chunk_size=1024,
        min_num_seqs=1,
        prefill_batch_threshold=16,
        max_prefill_num_seqs=16,
    )
    cfg.max_model_len = 65536
    cfg.use_chat_template = True
    return cfg


def _falcon3_7b_instruct_production_config():
    # Mirrors the real single-chip (p150) production launch config from
    # ~/scripts/model_servers/launch_falcon3_7b_instruct_uvicorn.sh (which
    # mirrors workflows/model_specs/dev/cnn.yaml), not the minimal smoke-test
    # defaults. Kept as the single test we maintain at production settings.
    cfg = _config(
        "tiiuae/Falcon3-7B-Instruct",
        32,
        gpu_memory_utilization=0.35,
        optimization_level=1,
        experimental_weight_dtype="bfp_bf8",
        experimental_kv_cache_dtype="bfp_bf8",
        enable_const_eval=True,
        min_context_len=128,
        prefill_chunk_size=1024,
        # b1-prefill: serve prefills serially (small graph) when <=16 are
        # pending instead of paying for a wasted-row b32 batch. Needs
        # min_num_seqs < max_num_seqs (batch_size=32).
        min_num_seqs=1,
        prefill_batch_threshold=16,
        # Cap prefill graphs at b16 instead of b32 (#5541): cuts prefill-trace
        # DRAM residency and ~22% off compile time, with no throughput/TTFT
        # regression measured.
        max_prefill_num_seqs=16,
    )
    # max_num_batched_tokens and enable_chunked_prefill aren't set here: the
    # plugin derives/overrides both from prefill_chunk_size (platform.py).
    cfg.max_model_len = 32768
    # Instruct-tuned: drive via the chat template, matching production evals
    # (--apply_chat_template).
    cfg.use_chat_template = True
    return cfg


SINGLE_DEVICE_CONFIGS = [
    # Llama
    pytest.param(_config("meta-llama/Llama-3.2-1B-Instruct"), id="llama-3.2-1b"),
    # BFP8 KV cache breaks n150 compile (L1-spill layout mismatch); see tt-mlir#9094.
    pytest.param(
        _config("meta-llama/Llama-3.2-3B-Instruct", experimental_kv_cache_dtype=""),
        id="llama-3.2-3b",
    ),
    pytest.param(
        _llama3_2_3b_instruct_production_config(),
        id="llama-3.2-3b-instruct-production",
    ),
    # opt-level 2 (the default since #5410) OOMs DRAM on n150. See https://github.com/tenstorrent/tt-xla/issues/5494.
    pytest.param(_config("meta-llama/Llama-3.1-8B-Instruct"), id="llama-3.1-8b"),
    pytest.param(
        _llama3_1_8b_instruct_production_config(),
        id="llama-3.1-8b-instruct-production",
    ),
    # Qwen 2.5
    pytest.param(_config("Qwen/Qwen2.5-0.5B-Instruct"), id="qwen2.5-0.5b-instruct"),
    pytest.param(_config("Qwen/Qwen2.5-1.5B-Instruct"), id="qwen2.5-1.5b-instruct"),
    pytest.param(_config("Qwen/Qwen2.5-3B-Instruct"), id="qwen2.5-3b-instruct"),
    pytest.param(_config("Qwen/Qwen2.5-7B-Instruct"), id="qwen2.5-7b-instruct"),
    # Qwen 3
    pytest.param(_config("Qwen/Qwen3-0.6B"), id="qwen3-0.6b"),
    pytest.param(_config("Qwen/Qwen3-1.7B"), id="qwen3-1.7b"),
    pytest.param(_config("Qwen/Qwen3-4B"), id="qwen3-4b"),
    pytest.param(_qwen3_4b_production_config(), id="qwen3-4b-production"),
    pytest.param(_config("Qwen/Qwen3-8B"), id="qwen3-8b"),
    pytest.param(_qwen3_8b_production_config(), id="qwen3-8b-production"),
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
    # BFP8 KV cache breaks n150 compile (L1-spill layout mismatch); see tt-mlir#9094.
    pytest.param(
        _config("tiiuae/Falcon3-3B-Base", experimental_kv_cache_dtype=""),
        id="falcon3-3b-base",
    ),
    pytest.param(
        _config("tiiuae/Falcon3-7B-Base", experimental_kv_cache_dtype=""),
        id="falcon3-7b-base",
    ),
    pytest.param(
        _falcon3_7b_instruct_production_config(),
        id="falcon3-7b-instruct-production",
    ),
    # Mistral
    pytest.param(
        _config("mistralai/Mistral-7B-Instruct-v0.3"), id="mistral-7b-instruct"
    ),
    pytest.param(
        _config(
            "mistralai/Ministral-8B-Instruct-2410",
            gpu_memory_utilization=0.12,
        ),
        id="ministral-8b",
    ),
    # OPT (vLLM-only fast canary; not part of the torch-xla matrix)
    pytest.param(_config("facebook/opt-125m"), id="opt-125m"),
]


# TP configs run exclusively on qb2-blackhole: 1D mesh, mesh_shape=None,
# auto-sized to the machine's device count.
TP_CONFIGS = [
    pytest.param(_tp_config("Qwen/Qwen3-32B", 32), id="qwen3-32b-qb2-tp"),
    pytest.param(_tp_config("tiiuae/Falcon3-7B-Base", 32), id="falcon3-7b-qb2-tp"),
    pytest.param(_tp_config("tiiuae/Falcon3-10B-Base", 32), id="falcon3-10b-qb2-tp"),
    pytest.param(
        _tp_config("Qwen/Qwen2.5-Coder-32B-Instruct", 32),
        id="qwen2.5-coder-32b-instruct-qb2-tp",
    ),
    pytest.param(
        _tp_config("mistralai/Mistral-Small-24B-Instruct-2501", 32),
        id="mistral-small-24b-instruct-2501-qb2-tp",
    ),
    pytest.param(
        _mistral_small_31_tp_config(
            "mistralai/Mistral-Small-3.1-24B-Instruct-2503", 32
        ),
        id="mistral-small-3.1-24b-tp",
    ),
    pytest.param(
        _tp_config("meta-llama/Llama-3.1-8B-Instruct", 32),
        id="llama-3.1-8b-qb2-tp",
    ),
    pytest.param(
        _tp_config(
            "meta-llama/Llama-3.1-70B-Instruct",
            32,
            gpu_memory_utilization=0.15,
            enable_const_eval=True,
            experimental_weight_dtype="bfp_bf8",
        ),
        id="llama-3.1-70b-qb2-tp",
    ),
    pytest.param(_gemma4_tp_config("google/gemma-4-31B-it", 32), id="gemma4-31b-it-tp"),
    # Verify fused decode_postprocess compiles to expected graph count (cpu_sampling=False path)
    pytest.param(
        _config("facebook/opt-125m", 1, gpu_memory_utilization=0.001),
        id="opt-125m-fused-measure",
    ),
    # Devstral-2-123B on the BH-galaxy (32 chips) in a 4x8 DP+TP layout:
    # mesh_shape=[4, 8] -> (batch=4 DP, model=8 TP). The 123B does not fit in a
    # TP-4 weight slice, so it needs model=8 (Qwen3-32B uses 8x4 instead; see
    # the qwen3-32b-bhglx branch). fp8 checkpoint -> bf16 via the dequant hook,
    # then stored as bfp8. shard_weights_on_batch_axis=False -> weights are
    # replicated across the 4 DP replicas (classic DP+TP, fewer CCLs). batch 32
    # -> 8 sequences/replica. Run with TT_RUNTIME_USING_BH_GALAXY=1.
    pytest.param(
        _tp_config(
            "mistralai/Devstral-2-123B-Instruct-2512",
            128,
            mesh_shape=[4, 8],
            experimental_weight_dtype="bfp_bf8",
            gpu_memory_utilization=0.3,
            enable_data_parallel=True,
            shard_weights_on_batch_axis=False,
        ),
        id="devstral-123b-galaxy-tp",
    ),
    # Qwen3-32B DP+TP on the 8x4 BH galaxy (DP=8, TP=4), batch 256 -> 32
    # sequences/replica (same per-replica load as devstral 128/4). Classic DP+TP
    # (shard_weights_on_batch_axis=False: weights replicated across the 8 DP
    # replicas, TP-sharded on the size-4 model axis); bf16 checkpoint stored as
    # bfp8 (no fp8 dequant hook). GMU 0.45: scaled from the validated batch-32
    # 8x4 config (0.0625 @ 4 seq/replica) -> ~0.43 @ 32 seq/replica; matches the
    # devstral-derived KV model (Qwen3 per-device KV ~1.45x: TP=4 keeps 2
    # kv-heads/device vs devstral's 1, over 64 vs 88 layers).
    # Run with TT_RUNTIME_USING_BH_GALAXY=1.
    pytest.param(
        _tp_config(
            "Qwen/Qwen3-32B",
            256,
            mesh_shape=[8, 4],
            experimental_weight_dtype="bfp_bf8",
            gpu_memory_utilization=0.45,
            enable_const_eval=True,
            enable_data_parallel=True,
            shard_weights_on_batch_axis=False,
        ),
        id="qwen3-32b-galaxy-tp",
    ),
]


def _run_vllm_benchmark(config, output_file, request, accuracy_testing=False):
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

    results = benchmark_vllm(config, display_name, accuracy_testing=accuracy_testing)

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


EMBEDDING_CONFIGS = [
    # Trace disabled: host/device tensor shape mismatch
    # (https://github.com/tenstorrent/tt-xla/issues/3936)
    pytest.param(
        _embedding_config(
            "Qwen/Qwen3-Embedding-4B", 1, max_model_len=128, enable_trace=False
        ),
        id="qwen3-embedding-4b-batch1",
    ),
    pytest.param(_embedding_config("BAAI/bge-m3", 1), id="bge-m3-batch1"),
    pytest.param(_embedding_config("BAAI/bge-m3", 32), id="bge-m3-batch32"),
]


def _dp_embedding_config(
    model: str,
    batch_size: int = 32,
    *,
    max_model_len: int = 512,
    gpu_memory_utilization: float = 0.05,
    optimization_level: int = 1,
    experimental_weight_dtype: str = "",
    enable_trace: bool = True,
    **additional_config_extra,
):
    additional = {
        "enable_data_parallel": True,
        "optimization_level": optimization_level,
        "enable_trace": enable_trace,
    }
    if experimental_weight_dtype:
        additional["experimental_weight_dtype"] = experimental_weight_dtype
    additional.update(additional_config_extra)
    return VLLMEmbeddingBenchmarkConfig(
        model=model,
        batch_size=batch_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        additional_config=additional,
    )


DP_EMBEDDING_CONFIGS = [
    pytest.param(
        _dp_embedding_config(
            "Qwen/Qwen3-Embedding-0.6B",
            32,
            max_model_len=128,
            experimental_weight_dtype="bfp_bf8",
        ),
        id="qwen3-embedding-0.6b-dp-batch32",
    ),
    pytest.param(
        _dp_embedding_config(
            "Qwen/Qwen3-Embedding-4B",
            32,
            max_model_len=128,
            experimental_weight_dtype="bfp_bf8",
        ),
        id="qwen3-embedding-4b-dp-batch32",
    ),
    pytest.param(
        _dp_embedding_config("BAAI/bge-m3", 32, experimental_weight_dtype="bfp_bf8"),
        id="bge-m3-dp-batch32",
    ),
]


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


@pytest.mark.parametrize("config", SINGLE_DEVICE_CONFIGS)
def test_vllm_benchmark(config, output_file, request, accuracy_testing):
    _run_vllm_benchmark(config, output_file, request, accuracy_testing=accuracy_testing)


@pytest.mark.parametrize("config", TP_CONFIGS)
def test_vllm_tp_benchmark(config, output_file, request, accuracy_testing):
    _run_vllm_benchmark(config, output_file, request, accuracy_testing=accuracy_testing)


@pytest.mark.parametrize("config", EMBEDDING_CONFIGS)
def test_vllm_embedding_benchmark(config, output_file, request):
    _run_vllm_embedding_benchmark(config, output_file, request)


@pytest.mark.parametrize("config", DP_EMBEDDING_CONFIGS)
def test_vllm_embedding_dp_benchmark(config, output_file, request):
    _run_vllm_embedding_benchmark(config, output_file, request)
