# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import socket
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import vllm
from utils import (
    create_benchmark_result,
    get_benchmark_metadata,
    print_benchmark_results,
)

DEFAULT_PROMPT = (
    "Here is an exhaustive list of the best practices for writing clean code:"
)


@dataclass
class VLLMBenchmarkConfig:
    """Configuration for a vLLM benchmark run."""

    # vLLM engine args
    model: str = "facebook/opt-125m"
    max_model_len: int = 128
    gpu_memory_utilization: float = 0.002

    # Per-modality input caps; zero them (e.g. {"image": 0, "video": 0,
    # "audio": 0}) to run a multimodal model text-only so its encoder never compiles.
    limit_mm_per_prompt: Optional[Dict[str, int]] = None

    # Floor for max_num_batched_tokens (effective = max(batch * len, this)).
    # Some multimodal models need a higher floor (Gemma-4: >= 2560 for the
    # MultiModalBudget video-frame floor).
    min_num_batched_tokens: int = 0

    # Explicit per-step token budget. When set, overrides the derived
    # batch * max_model_len value, decoupling the budget from max_model_len so
    # chunked prefill (tt-xla #4986) can cap precompile buckets + prefill DRAM.
    max_num_batched_tokens: Optional[int] = None

    # TT compile options passed directly to vLLM's additional_config (TTConfig).
    additional_config: Dict[str, Any] = field(default_factory=dict)

    # Benchmark params
    batch_size: int = 1
    max_tokens: int = 128
    warmup_iterations: int = 1
    prompt: str = DEFAULT_PROMPT

    # Exact input length via dummy prompt_token_ids (overrides `prompt` when > 0).
    # Lets us test a large ISL (e.g. 16384) that a short text prompt can't reach.
    isl: int = 0

    # When True, send `prompt` as a chat message via llm.chat() so the chat
    # template is applied; instruct models (e.g. Gemma-4-it) degenerate without it.
    use_chat_template: bool = False

    # Sampling params (temperature=0.0 -> greedy)
    temperature: float = 0.0


@dataclass
class VLLMEmbeddingBenchmarkConfig:
    """Configuration for a vLLM embedding benchmark run."""

    model: str = "BAAI/bge-m3"
    max_model_len: int = 512
    gpu_memory_utilization: float = 0.05
    additional_config: Dict[str, Any] = field(default_factory=dict)
    batch_size: int = 1
    warmup_iterations: int = 1
    loop_count: int = 32


def _create_llm(config: VLLMBenchmarkConfig) -> vllm.LLM:
    """Build engine args from config and create a vLLM LLM instance."""
    additional_config = dict(config.additional_config)
    # Default to device sampling; opt-in to CPU sampling per-config when
    # needed (e.g. via the TT_BENCHMARK_CPU_SAMPLING env var in
    # tests/benchmark/test_vllm_benchmarks.py).
    additional_config.setdefault("cpu_sampling", False)

    if config.max_num_batched_tokens is not None:
        max_num_batched_tokens = config.max_num_batched_tokens
    else:
        max_num_batched_tokens = max(
            config.batch_size * config.max_model_len, config.min_num_batched_tokens
        )

    llm_args: Dict[str, Any] = {
        "model": config.model,
        "max_model_len": config.max_model_len,
        "max_num_seqs": config.batch_size,
        "max_num_batched_tokens": max_num_batched_tokens,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "disable_log_stats": False,
        "additional_config": additional_config,
    }
    if config.limit_mm_per_prompt is not None:
        llm_args["limit_mm_per_prompt"] = config.limit_mm_per_prompt

    print(f"Creating vLLM engine for {config.model} ...")
    print(f"  LLM args: {llm_args}")
    print(f"  Sampling params: max_tokens={config.max_tokens}, ignore_eos=True")
    print(f"  Batch size (num prompts): {config.batch_size}")

    return vllm.LLM(**llm_args)


def _extract_metrics(
    outputs: List[vllm.RequestOutput],
) -> Tuple[float, List[int], float, float]:
    """
    Extract per-request metrics and return aggregated per-user values.

    Returns:
        (avg_ttft_ms, tokens_per_user, avg_decode_time_s, avg_tokens_per_sec)
    """
    ttft_values = []

    for i, output in enumerate(outputs):
        stats = output.metrics
        gen_tokens = len(output.outputs[0].token_ids)

        ttft_ms = stats.first_token_latency * 1000.0
        ttft_values.append(ttft_ms)

        decode_tokens = stats.num_generation_tokens - 1
        decode_time = stats.last_token_ts - stats.first_token_ts
        if decode_time > 0 and decode_tokens > 0:
            tps = decode_tokens / decode_time
            print(
                f"  Request {i}: gen_tokens={gen_tokens}, "
                f"TTFT={ttft_ms:.1f}ms, "
                f"decode_tokens={decode_tokens}, "
                f"decode_time={decode_time:.3f}s, "
                f"decode_tps={tps:.1f}"
            )
        else:
            print(f"  Request {i}: gen_tokens={gen_tokens}, TTFT={ttft_ms:.1f}ms")

    avg_ttft_ms = sum(ttft_values) / len(ttft_values) if ttft_values else 0.0

    tokens_per_user = [len(o.outputs[0].token_ids) - 1 for o in outputs]
    first_token_times = [o.metrics.first_token_ts for o in outputs]
    last_token_times = [o.metrics.last_token_ts for o in outputs]

    decode_times_per_user = [
        last_token_time - first_token_time
        for last_token_time, first_token_time in zip(
            last_token_times, first_token_times
        )
    ]
    tokens_per_sec_per_user = [
        tokens / decode_time if decode_time > 0 else 0.0
        for tokens, decode_time in zip(tokens_per_user, decode_times_per_user)
    ]
    avg_tokens_per_sec = sum(tokens_per_sec_per_user) / len(tokens_per_sec_per_user)
    avg_decode_time_s = (
        sum(decode_times_per_user) / len(decode_times_per_user)
        if decode_times_per_user
        else 0.0
    )

    return (
        avg_ttft_ms,
        tokens_per_user,
        avg_decode_time_s,
        avg_tokens_per_sec,
    )


def _get_device_info(
    config: VLLMBenchmarkConfig,
) -> Tuple[str, int, Optional[Tuple[int, int]]]:
    """
    Derive device info from config.

    This is a workaround as these info are needed for the benchmark schema, but
    vLLM abstracts the device layer. Mesh shape follows the plugin convention (num_devices, 1).

    Returns:
        (arch, device_count, mesh_shape)
    """
    if config.additional_config.get("enable_tensor_parallel", False):
        return "wormhole_llmbox", 8, (8, 1)
    return "wormhole", 1, None


def _assert_token_counts(
    outputs: List[vllm.RequestOutput], max_tokens: int, max_model_len: int
):
    """Assert every request generated the expected number of tokens."""
    for i, output in enumerate(outputs):
        prompt_len = len(output.prompt_token_ids)
        expected = min(max_tokens, max_model_len - prompt_len)
        actual = len(output.outputs[0].token_ids)
        assert actual == expected, (
            f"Request {i} generated {actual} tokens, expected {expected} "
            f"(prompt_len={prompt_len}, max_tokens={max_tokens}, "
            f"max_model_len={max_model_len}). "
            f"This may indicate preemption or OOM."
        )


def _assert_no_preemptions(llm: vllm.LLM):
    """
    Assert the engine had zero preemptions during the run.

    Failing this assertion usually means more memory is needed for the KV Cache,
    which can be adjusted through the gpu_memory_utilization config field.
    """
    for metric in llm.get_metrics():
        if metric.name == "vllm:num_preemptions":
            assert metric.value == 0, (
                f"Preemptions detected: {metric.value}. "
                "KV Cache size likely needs to be increased."
            )
            return
    assert False, "vllm:num_preemptions metric not found in engine metrics."


def benchmark_vllm(
    config: VLLMBenchmarkConfig,
    display_name: str,
) -> Dict[str, Any]:
    """Run a vLLM benchmark and return a standardised result dict."""
    sampling_params = vllm.SamplingParams(
        max_tokens=config.max_tokens,
        ignore_eos=True,
        temperature=config.temperature,
    )

    llm = _create_llm(config)

    # chat() applies the model's chat template; generate() feeds the raw
    # prompt. Same (inputs, sampling_params) -> List[RequestOutput] signature.
    if config.isl > 0:
        # Exact-length input via dummy token ids (100 is a safe in-vocab id).
        inputs = [
            {"prompt_token_ids": [100] * config.isl} for _ in range(config.batch_size)
        ]
        run_fn = llm.generate
    elif config.use_chat_template:
        inputs = [
            [{"role": "user", "content": config.prompt}]
            for _ in range(config.batch_size)
        ]
        run_fn = llm.chat
    else:
        inputs = [config.prompt] * config.batch_size
        run_fn = llm.generate

    def _run() -> List[vllm.RequestOutput]:
        return run_fn(inputs, sampling_params)

    if config.warmup_iterations > 0:
        print(f"\nWarming up ({config.warmup_iterations} iteration(s)) ...")
        for _ in range(config.warmup_iterations):
            _run()
        print("Warmup complete.")

    print(f"\nStarting benchmark ({config.max_tokens} tokens) ...")
    outputs: List[vllm.RequestOutput] = _run()

    # Print generated text for output-quality inspection.
    for i, output in enumerate(outputs):
        print(f"  [{i}] {output.prompt!r} -> {output.outputs[0].text!r}")

    # Assert decode is consistent
    _assert_token_counts(outputs, config.max_tokens, config.max_model_len)
    _assert_no_preemptions(llm)

    (
        avg_ttft_ms,
        tokens_per_user,
        avg_decode_time_s,
        avg_tokens_per_sec,
    ) = _extract_metrics(outputs)
    total_samples = sum(tokens_per_user)

    metadata = get_benchmark_metadata()
    full_model_name = config.model
    model_type = "text-generation"
    dataset_name = "Random Data"
    # vLLM doesn't expose raw logits, so PCC comparison is not possible.
    evaluation_score = 0.0
    custom_measurements = [
        {
            "measurement_name": "ttft",
            "value": avg_ttft_ms,
            "target": -1,
        },
        {
            "measurement_name": "samples_per_sec",
            "value": avg_tokens_per_sec,
            "target": -1,
        },
    ]

    print_benchmark_results(
        model_title=full_model_name,
        full_model_name=full_model_name,
        model_type=model_type,
        dataset_name=dataset_name,
        date=metadata["date"],
        machine_name=metadata["machine_name"],
        total_time=avg_decode_time_s,
        total_samples=total_samples,
        samples_per_sec=avg_tokens_per_sec,
        evaluation_score=evaluation_score,
        batch_size=config.batch_size,
        data_format="bfloat16",
        input_sequence_length=config.max_model_len,
        ttft_ms=avg_ttft_ms,
    )

    arch, device_count, mesh_shape = _get_device_info(config)

    return create_benchmark_result(
        full_model_name=full_model_name,
        model_type=model_type,
        dataset_name=dataset_name,
        num_layers=-1,
        batch_size=config.batch_size,
        input_size=(config.max_model_len,),
        loop_count=1,
        data_format="bfloat16",
        total_time=avg_decode_time_s,
        total_samples=total_samples,
        evaluation_score=evaluation_score,
        custom_measurements=custom_measurements,
        optimization_level=config.additional_config.get("optimization_level", 0),
        program_cache_enabled=True,
        trace_enabled=config.additional_config.get("enable_trace", False),
        experimental_weight_dtype=(
            "bfp_bf8"
            if config.additional_config.get(
                "experimental_enable_weight_bfp8_conversion", False
            )
            else ""
        ),
        model_info=full_model_name,
        display_name=display_name,
        torch_xla_enabled=True,
        backend="tt",
        device_name=socket.gethostname(),
        arch=arch,
        input_is_image=False,
        input_sequence_length=config.max_model_len,
        device_count=device_count,
        mesh_shape=mesh_shape,
        vllm=True,
    )


def benchmark_vllm_embedding(
    config: VLLMEmbeddingBenchmarkConfig,
    display_name: str,
) -> Dict[str, Any]:
    """Run a vLLM embedding benchmark and return a standardised result dict."""
    prompts = [DEFAULT_PROMPT] * config.batch_size

    llm_args: Dict[str, Any] = {
        "model": config.model,
        "max_model_len": config.max_model_len,
        "max_num_seqs": config.batch_size,
        "max_num_batched_tokens": config.batch_size * config.max_model_len,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "disable_log_stats": False,
        "additional_config": dict(config.additional_config),
    }
    print(f"Creating vLLM embedding engine for {config.model} ...")
    print(f"  LLM args: {llm_args}")
    llm = vllm.LLM(**llm_args)

    if config.warmup_iterations > 0:
        print(f"\nWarming up ({config.warmup_iterations} iteration(s)) ...")
        for _ in range(config.warmup_iterations):
            llm.embed(prompts)
        print("Warmup complete.")

    print(f"\nStarting benchmark ({config.loop_count} iterations) ...")
    total_time = 0.0
    for _ in range(config.loop_count):
        t0 = time.perf_counter()
        llm.embed(prompts)
        total_time += time.perf_counter() - t0

    avg_time = total_time / config.loop_count
    samples_per_sec = config.batch_size / avg_time

    metadata = get_benchmark_metadata()
    full_model_name = config.model
    model_type = "text-embedding"
    dataset_name = "Random Data"
    evaluation_score = 0.0

    print_benchmark_results(
        model_title=full_model_name,
        full_model_name=full_model_name,
        model_type=model_type,
        dataset_name=dataset_name,
        date=metadata["date"],
        machine_name=metadata["machine_name"],
        total_time=avg_time,
        total_samples=config.batch_size,
        samples_per_sec=samples_per_sec,
        evaluation_score=evaluation_score,
        batch_size=config.batch_size,
        input_sequence_length=config.max_model_len,
    )

    return create_benchmark_result(
        full_model_name=full_model_name,
        model_type=model_type,
        dataset_name=dataset_name,
        num_layers=-1,
        batch_size=config.batch_size,
        input_size=(config.max_model_len,),
        loop_count=config.loop_count,
        data_format="bfloat16",
        total_time=avg_time,
        total_samples=config.batch_size,
        evaluation_score=evaluation_score,
        optimization_level=config.additional_config.get("optimization_level", 0),
        program_cache_enabled=True,
        trace_enabled=config.additional_config.get("enable_trace", True),
        model_info=full_model_name,
        display_name=display_name,
        torch_xla_enabled=True,
        backend="tt",
        device_name=socket.gethostname(),
        arch="wormhole",
        input_is_image=False,
        input_sequence_length=config.max_model_len,
        device_count=1,
        vllm=True,
    )
