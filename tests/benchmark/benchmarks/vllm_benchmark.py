# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import socket
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

    # TT compile options passed directly to vLLM's additional_config (TTConfig).
    additional_config: Dict[str, Any] = field(default_factory=dict)

    # Benchmark params
    batch_size: int = 1
    max_tokens: int = 128
    warmup_iterations: int = 1


def _create_llm(config: VLLMBenchmarkConfig) -> vllm.LLM:
    """Build engine args from config and create a vLLM LLM instance."""
    additional_config = dict(config.additional_config)
    additional_config.setdefault("cpu_sampling", True)

    llm_args: Dict[str, Any] = {
        "model": config.model,
        "max_model_len": config.max_model_len,
        "max_num_seqs": config.batch_size,
        "max_num_batched_tokens": config.batch_size * config.max_model_len,
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
) -> Tuple[float, float, int, float]:
    """Extract per-request metrics and return aggregated values.

    Returns:
        (avg_ttft_ms, tokens_per_second, decode_total_tokens, decode_total_time)
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
    first_token_times = [o.metrics.first_token_ts for o in outputs]
    last_token_times = [o.metrics.last_token_ts for o in outputs]
    decode_total_time = max(last_token_times) - min(first_token_times)
    tokens_per_second = (
        (decode_total_tokens / decode_total_time) if decode_total_time > 0 else 0.0
    )

    return avg_ttft_ms, tokens_per_second, decode_total_tokens, decode_total_time


def _get_device_info(
    config: VLLMBenchmarkConfig,
) -> Tuple[str, int, Optional[Tuple[int, int]]]:
    """Derive device info from config.

    This is a workaround as these info are needed for the benchmark schema, but
    vLLM abstracts the device layer. Mesh shape follows the plugin convention (num_devices, 1).

    Returns:
        (arch, device_count, mesh_shape)
    """
    if config.additional_config.get("enable_tensor_parallel", False):
        return "wormhole_llmbox", 8, (8, 1)
    return "wormhole", 1, None


def benchmark_vllm(
    config: VLLMBenchmarkConfig,
    display_name: str,
) -> Dict[str, Any]:
    """Run a vLLM benchmark and return a standardised result dict."""
    prompts = [DEFAULT_PROMPT] * config.batch_size
    sampling_params = vllm.SamplingParams(max_tokens=config.max_tokens, ignore_eos=True)

    llm = _create_llm(config)

    if config.warmup_iterations > 0:
        print(f"\nWarming up ({config.warmup_iterations} iteration(s)) ...")
        warmup_params = vllm.SamplingParams(max_tokens=1, ignore_eos=True)
        for _ in range(config.warmup_iterations):
            llm.generate(prompts, warmup_params)
        print("Warmup complete.")

    print(f"\nStarting benchmark ({config.max_tokens} tokens) ...")
    outputs: List[vllm.RequestOutput] = llm.generate(prompts, sampling_params)

    avg_ttft_ms, tokens_per_second, decode_total_tokens, decode_total_time = (
        _extract_metrics(outputs, config.batch_size)
    )

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
        total_samples=decode_total_tokens,
        samples_per_sec=tokens_per_second,
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
        total_samples=decode_total_tokens,
        evaluation_score=evaluation_score,
        custom_measurements=custom_measurements,
        optimization_level=config.additional_config.get("optimization_level", 0),
        program_cache_enabled=True,
        trace_enabled=False,
        enable_weight_bfp8_conversion=config.additional_config.get(
            "experimental_enable_weight_bfp8_conversion", False
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
    )
