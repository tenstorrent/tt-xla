# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import socket
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import vllm
from utils import (
    align_arch,
    create_benchmark_result,
    get_benchmark_metadata,
    print_benchmark_results,
)

# Accuracy testing: total reference length, split in half by
# init_accuracy_testing() into prompt (prefill) and teacher-forced decode.
# Must stay equal to test_llms.py's DEFAULT_INPUT_SEQUENCE_LENGTH: both paths
# share the same reference_outputs/<model>.refpt files and needs_regeneration()
# does not compare total_length, so divergence silently scores one path against
# a mismatched context window.
_ACCURACY_TOTAL_LENGTH = 128

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

    # TT compile options passed directly to vLLM's additional_config (TTConfig).
    additional_config: Dict[str, Any] = field(default_factory=dict)

    # Benchmark params
    batch_size: int = 1
    max_tokens: int = 128
    warmup_iterations: int = 1
    prompt: str = DEFAULT_PROMPT

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


def _get_device_info_from_engine(
    llm: vllm.LLM,
) -> Tuple[str, int, Optional[Tuple[int, int]]]:
    """
    Read real TT device info from the live vLLM engine's worker(s).

    Returns:
        (arch, device_count, mesh_shape)
    """
    arch = ""
    device_count = 1
    mesh_shape = None
    try:
        results = llm.collective_rpc("get_device_info")
        if results and results[0]:
            info = results[0]
            arch = align_arch(str(info.get("arch", "")).lower())
            device_count = max(int(info.get("device_count", 1)), 1)
            resolved_mesh = info.get("mesh_shape")
            if isinstance(resolved_mesh, (list, tuple)) and len(resolved_mesh) == 2:
                mesh_shape = (int(resolved_mesh[0]), int(resolved_mesh[1]))
    except Exception as e:
        print(
            f"Warning: could not read TT device info from engine ({e}); using defaults."
        )

    return arch, max(int(device_count), 1), mesh_shape


def _create_llm(config: VLLMBenchmarkConfig) -> vllm.LLM:
    """Build engine args from config and create a vLLM LLM instance."""
    additional_config = dict(config.additional_config)
    # Default to device sampling; opt-in to CPU sampling per-config when
    # needed (e.g. via the TT_BENCHMARK_CPU_SAMPLING env var in
    # tests/benchmark/test_vllm_benchmarks.py).
    additional_config.setdefault("cpu_sampling", False)

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


def _extract_decode_predictions(
    output: vllm.RequestOutput, expected_count: int
) -> List[int]:
    """Per-step device argmax token ids from a teacher-forced accuracy run.

    With teacher forcing, `output.outputs[0].token_ids` are the injected
    ground-truth tokens, so the device's own predictions must be read from the
    per-step sample logprobs (each step's highest-logprob token is the device
    argmax). Asserts the full decode window is present so a backend that
    silently drops logprobs fails loudly instead of reporting a false 0%
    accuracy regression.
    """
    step_logprobs = output.outputs[0].logprobs
    assert step_logprobs is not None, (
        "No decode logprobs returned; the vLLM backend must honor "
        "SamplingParams(logprobs=1) for accuracy testing."
    )
    assert len(step_logprobs) == expected_count, (
        f"Expected {expected_count} decode-step logprobs, got "
        f"{len(step_logprobs)}. Backend returned an incomplete decode window."
    )
    predicted_tokens = []
    for pos_logprobs in step_logprobs:
        top1_token = max(pos_logprobs, key=lambda k: pos_logprobs[k].logprob)
        predicted_tokens.append(top1_token)
    return predicted_tokens


def _perf_measurements(avg_ttft_ms: float, avg_tokens_per_sec: float) -> List[dict]:
    return [
        {"measurement_name": "ttft", "value": avg_ttft_ms, "target": -1},
        {
            "measurement_name": "samples_per_sec",
            "value": avg_tokens_per_sec,
            "target": -1,
        },
    ]


def _benchmark_vllm_accuracy(
    config: VLLMBenchmarkConfig,
    display_name: str,
) -> Dict[str, Any]:
    """Teacher-forced decode accuracy run through vLLM.

    Prompts with the prefill half of the reference sequence and generates the
    decode half; the vllm_tt runner overrides each sampled token with the
    ground-truth token (via extra_args), so every decode step sees the reference
    context — exactly like llm_benchmark.py. The device's own argmax per step
    survives in `logprobs` (gathered before the override), which is what gets
    scored. Prediction 0 comes from the prefill forward; predictions 1..N-1 come
    from the decode kernel.
    """
    # Imported here rather than at module scope: llm_utils.decode_utils and
    # transformers pull in heavy deps (decode_utils imports tracy / infra /
    # tt_torch, hence torch_xla). Perf runs use --confcutdir=tests/benchmark
    # specifically so torch_xla never enters the vLLM engine process.
    from llm_utils.decode_utils import init_accuracy_testing
    from llm_utils.token_accuracy import score_token_accuracy
    from transformers import AutoTokenizer

    # Shares the .refpt on-demand regeneration + half/half prefill/decode split
    # with the custom torch-xla path (tests/benchmark/test_llms.py), so both
    # paths score against identically-built reference data.
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    token_accuracy, _ = init_accuracy_testing(
        model_name_for_accuracy=config.model.split("/")[-1],
        max_cache_len=_ACCURACY_TOTAL_LENGTH,
        tokenizer=tokenizer,
        hf_model_name=config.model,
    )

    llm = _create_llm(config)
    arch, device_count, mesh_shape = _get_device_info_from_engine(llm)

    input_prompt_tokens = token_accuracy.input_prompt.tolist()
    ground_truth_tokens = token_accuracy.reference_tokens.tolist()
    num_decode = len(ground_truth_tokens)

    accuracy_params = vllm.SamplingParams(
        max_tokens=num_decode,
        ignore_eos=True,
        temperature=0.0,
        logprobs=1,
        extra_args={"teacher_forcing_tokens": ground_truth_tokens},
    )
    print(
        f"\nRunning accuracy test (prompt={len(input_prompt_tokens)} tokens, "
        f"teacher-forced decode={num_decode} tokens, "
        f"batch_size={config.batch_size})..."
    )
    accuracy_outputs = llm.generate(
        [{"prompt_token_ids": input_prompt_tokens} for _ in range(config.batch_size)],
        accuracy_params,
    )

    # A preempted request re-prefills and loses its teacher-forced context,
    # which shows up as an accuracy drop that looks like a model regression.
    _assert_no_preemptions(llm)

    # The accuracy run is a real generate, so report perf alongside accuracy
    # rather than leaving zeros in the record, matching llm_benchmark.py.
    (
        avg_ttft_ms,
        tokens_per_user,
        avg_decode_time_s,
        avg_tokens_per_sec,
    ) = _extract_metrics(accuracy_outputs)

    per_user_predictions = [
        _extract_decode_predictions(output, num_decode) for output in accuracy_outputs
    ]

    # Eyeball check on user 0: both streams are per-position teacher-forced
    # argmax (not free-running text), so the fraction of matching tokens equals
    # TOP1. Clamped to the window compute_accuracy() actually scores.
    golden_ids = token_accuracy.top1_tokens.tolist()
    scored = min(len(golden_ids), len(per_user_predictions[0]))
    print(f"\n  [golden] {tokenizer.decode(golden_ids[:scored])!r}")
    print(f"  [device] {tokenizer.decode(per_user_predictions[0][:scored])!r}")

    evaluation_score, accuracy_measurements = score_token_accuracy(
        token_accuracy, per_user_predictions
    )

    return _build_result(
        config=config,
        display_name=display_name,
        arch=arch,
        device_count=device_count,
        mesh_shape=mesh_shape,
        dataset_name="Tale of Two Cities (Reference Data)",
        evaluation_score=evaluation_score,
        avg_ttft_ms=avg_ttft_ms,
        avg_decode_time_s=avg_decode_time_s,
        avg_tokens_per_sec=avg_tokens_per_sec,
        total_samples=sum(tokens_per_user),
        custom_measurements=_perf_measurements(avg_ttft_ms, avg_tokens_per_sec)
        + accuracy_measurements,
    )


def benchmark_vllm(
    config: VLLMBenchmarkConfig,
    display_name: str,
    accuracy_testing: bool = False,
) -> Dict[str, Any]:
    """Run a vLLM benchmark and return a standardised result dict."""
    if accuracy_testing:
        return _benchmark_vllm_accuracy(config, display_name)

    sampling_params = vllm.SamplingParams(
        max_tokens=config.max_tokens,
        ignore_eos=True,
        temperature=config.temperature,
    )

    llm = _create_llm(config)
    arch, device_count, mesh_shape = _get_device_info_from_engine(llm)

    # chat() applies the model's chat template; generate() feeds the raw
    # prompt. Same (inputs, sampling_params) -> List[RequestOutput] signature.
    if config.use_chat_template:
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

    return _build_result(
        config=config,
        display_name=display_name,
        arch=arch,
        device_count=device_count,
        mesh_shape=mesh_shape,
        dataset_name="Random Data",
        # vLLM doesn't expose raw logits, so PCC comparison is not possible.
        evaluation_score=0.0,
        avg_ttft_ms=avg_ttft_ms,
        avg_decode_time_s=avg_decode_time_s,
        avg_tokens_per_sec=avg_tokens_per_sec,
        total_samples=sum(tokens_per_user),
        custom_measurements=_perf_measurements(avg_ttft_ms, avg_tokens_per_sec),
    )


def _build_result(
    config: VLLMBenchmarkConfig,
    display_name: str,
    arch: str,
    device_count: int,
    mesh_shape: Any,
    dataset_name: str,
    evaluation_score: float,
    avg_ttft_ms: float,
    avg_decode_time_s: float,
    avg_tokens_per_sec: float,
    total_samples: int,
    custom_measurements: List[dict],
) -> Dict[str, Any]:
    """Print the benchmark summary and build the standardised result dict."""
    metadata = get_benchmark_metadata()
    full_model_name = config.model
    model_type = "text-generation"

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

    arch, device_count, _ = _get_device_info_from_engine(llm)

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
        arch=arch,
        input_is_image=False,
        input_sequence_length=config.max_model_len,
        device_count=device_count,
        vllm=True,
    )
