# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import socket
import statistics
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
    # Explicit override. None -> derived as batch_size * max_model_len.
    # Needed for models whose per-item multimodal token budget is larger than
    # batch_size * max_model_len (e.g. Gemma-4 video: 2496).
    max_num_batched_tokens: Optional[int] = None

    # TT compile options passed directly to vLLM's additional_config (TTConfig).
    additional_config: Dict[str, Any] = field(default_factory=dict)

    # Device metadata overrides for the benchmark result schema. When unset,
    # falls back to derivation from enable_tensor_parallel.
    arch: Optional[str] = None
    device_count: Optional[int] = None
    mesh_shape: Optional[Tuple[int, int]] = None

    # Benchmark params
    batch_size: int = 1
    max_tokens: int = 128
    warmup_iterations: int = 1
    # Number of measured benchmark iterations; >1 prints per-iteration stats
    # (mean / min / max / p15 / p50 / p85 / stdev) to characterize run-to-run
    # variation within a single test invocation.
    measured_iterations: int = 1


def _create_llm(config: VLLMBenchmarkConfig) -> vllm.LLM:
    """Build engine args from config and create a vLLM LLM instance."""
    additional_config = dict(config.additional_config)
    # Using CPU sampling so that we have batch_size = 32
    # See issue: https://github.com/tenstorrent/tt-xla/issues/3610
    additional_config.setdefault("cpu_sampling", True)

    max_num_batched_tokens = (
        config.max_num_batched_tokens
        if config.max_num_batched_tokens is not None
        else config.batch_size * config.max_model_len
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

    print(f"Creating vLLM engine for {config.model} ...")
    print(f"  LLM args: {llm_args}")
    print(f"  Sampling params: max_tokens={config.max_tokens}, ignore_eos=True")
    print(f"  Batch size (num prompts): {config.batch_size}")

    return vllm.LLM(**llm_args)


def _extract_metrics(
    outputs: List[vllm.RequestOutput],
    batch_size: int,
) -> Tuple[float, int, float, float]:
    """
    Extract per-request metrics and return aggregated per-user values.

    Returns:
        (avg_ttft_ms, tokens_per_user, decode_total_time, tokens_per_sec_per_user)
    """
    ttft_values = []
    total_gen_tokens = 0

    for i, output in enumerate(outputs):
        stats = output.metrics
        gen_tokens = len(output.outputs[0].token_ids)
        total_gen_tokens += gen_tokens

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

    decode_total_tokens = total_gen_tokens - batch_size
    tokens_per_user = decode_total_tokens // batch_size

    first_token_times = [o.metrics.first_token_ts for o in outputs]
    last_token_times = [o.metrics.last_token_ts for o in outputs]
    decode_total_time = max(last_token_times) - min(first_token_times)
    tokens_per_sec_per_user = (
        (tokens_per_user / decode_total_time) if decode_total_time > 0 else 0.0
    )

    return avg_ttft_ms, tokens_per_user, decode_total_time, tokens_per_sec_per_user


def _get_device_info(
    config: VLLMBenchmarkConfig,
) -> Tuple[str, int, Optional[Tuple[int, int]]]:
    """
    Derive device info from config.

    This is a workaround as these info are needed for the benchmark schema, but
    vLLM abstracts the device layer. Explicit overrides on the config win;
    otherwise fall back to defaults keyed on enable_tensor_parallel.

    Returns:
        (arch, device_count, mesh_shape)
    """
    if (
        config.arch is not None
        and config.device_count is not None
        and config.mesh_shape is not None
    ):
        return config.arch, config.device_count, config.mesh_shape

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


def _percentile(sorted_vals: List[float], pct: float) -> float:
    """Linear-interpolated percentile over an already-sorted list (ascending)."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _print_stats(label: str, values: List[float], unit: str) -> None:
    """Print mean / min / max / p15 / p50 / p85 / stdev for a sample list."""
    if not values:
        print(f"  {label}: <no samples>")
        return
    sorted_vals = sorted(values)
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    spread_pct = (sorted_vals[-1] - sorted_vals[0]) / mean * 100.0 if mean else 0.0
    print(
        f"  {label} ({unit}, n={len(values)}): "
        f"mean={mean:.2f} stdev={stdev:.2f} "
        f"min={sorted_vals[0]:.2f} max={sorted_vals[-1]:.2f} "
        f"p15={_percentile(sorted_vals, 15):.2f} "
        f"p50={_percentile(sorted_vals, 50):.2f} "
        f"p85={_percentile(sorted_vals, 85):.2f} "
        f"(max-min)/mean={spread_pct:.1f}%"
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
    prompts = [DEFAULT_PROMPT] * config.batch_size
    sampling_params = vllm.SamplingParams(
        max_tokens=config.max_tokens, ignore_eos=True, temperature=0.6
    )

    llm = _create_llm(config)

    if config.warmup_iterations > 0:
        print(f"\nWarming up ({config.warmup_iterations} iteration(s)) ...")
        for _ in range(config.warmup_iterations):
            llm.generate(prompts, sampling_params)
        print("Warmup complete.")

    n_iter = max(1, config.measured_iterations)
    print(
        f"\nStarting benchmark ({config.max_tokens} tokens, "
        f"{n_iter} measured iteration(s)) ..."
    )

    iter_ttft_ms: List[float] = []
    iter_tokens_per_user: List[int] = []
    iter_decode_time: List[float] = []
    iter_tps: List[float] = []
    per_request_tps: List[float] = []

    for it in range(n_iter):
        if n_iter > 1:
            print(f"\n--- Iteration {it + 1}/{n_iter} ---")
        outputs: List[vllm.RequestOutput] = llm.generate(prompts, sampling_params)

        _assert_token_counts(outputs, config.max_tokens, config.max_model_len)
        _assert_no_preemptions(llm)

        avg_ttft_ms, tokens_per_user, decode_total_time, tokens_per_sec_per_user = (
            _extract_metrics(outputs, config.batch_size)
        )
        iter_ttft_ms.append(avg_ttft_ms)
        iter_tokens_per_user.append(tokens_per_user)
        iter_decode_time.append(decode_total_time)
        iter_tps.append(tokens_per_sec_per_user)
        for o in outputs:
            stats = o.metrics
            decode_tokens = stats.num_generation_tokens - 1
            decode_time = stats.last_token_ts - stats.first_token_ts
            if decode_time > 0 and decode_tokens > 0:
                per_request_tps.append(decode_tokens / decode_time)

    if n_iter > 1:
        print("\n=== Aggregate stats across measured iterations ===")
        _print_stats("decode_tps (per-iter, per-user)", iter_tps, "tok/s")
        _print_stats("ttft", iter_ttft_ms, "ms")
        _print_stats("decode_total_time", iter_decode_time, "s")
        if config.batch_size > 1:
            _print_stats("decode_tps (per-request)", per_request_tps, "tok/s")
        print()

    avg_ttft_ms = statistics.fmean(iter_ttft_ms)
    tokens_per_user = iter_tokens_per_user[-1]
    decode_total_time = statistics.fmean(iter_decode_time)
    tokens_per_sec_per_user = statistics.fmean(iter_tps)

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
    ]

    print_benchmark_results(
        model_title=full_model_name,
        full_model_name=full_model_name,
        model_type=model_type,
        dataset_name=dataset_name,
        date=metadata["date"],
        machine_name=metadata["machine_name"],
        total_time=decode_total_time,
        total_samples=tokens_per_user,
        samples_per_sec=tokens_per_sec_per_user,
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
        total_time=decode_total_time,
        total_samples=tokens_per_user,
        evaluation_score=evaluation_score,
        custom_measurements=custom_measurements,
        optimization_level=config.additional_config.get("optimization_level", 0),
        program_cache_enabled=True,
        trace_enabled=False,
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
