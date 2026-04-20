# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Built-in modules
import os
import socket
import sys
from typing import Optional

import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
import tracy
import transformers
from llm_utils import generate_and_benchmark, init_accuracy_testing, init_static_cache
from llm_utils.decode_utils import LLMSamplingWrapper
from loguru import logger
from torch_xla.distributed.spmd import Mesh
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from transformers.cache_utils import StaticCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from tt_torch.sharding import sharding_constraint_hook
from tt_torch.weight_dtype import apply_weight_dtype_overrides
from utils import (
    build_xla_export_name,
    compute_pcc,
    create_benchmark_result,
    get_benchmark_metadata,
    get_xla_device_arch,
    print_benchmark_results,
)

xr.set_device_type("TT")

MIN_STEPS = 16

# Default input prompt
DEFAULT_INPUT_PROMPT = (
    "Here is an exaustive list of the best practices for writing clean code:"
)

MODULE_EXPORT_PATH = "modules"


def setup_model_and_tokenizer(
    model_loader, model_variant
) -> tuple[torch.nn.Module, PreTrainedTokenizer]:
    """
    Instantiate model and tokenizer.

    Args:
        model_loader: Loader of the HuggingFace model.
        model_variant: Specific variant of the model.

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model {model_loader.get_model_info(variant=model_variant).name}...")

    model = model_loader.load_model(dtype_override=torch.bfloat16)
    if hasattr(model.config, "layer_types"):
        model.config.layer_types = ["full_attention"] * len(model.config.layer_types)
    # Use static dense experts forward to avoid graph breaks from data-dependent
    # loops in the original experts and _grouped_mm CPU crashes.
    if hasattr(model.config, "_experts_implementation"):
        model.config._experts_implementation = "dense"
    model = model.eval()
    tokenizer = model_loader.tokenizer

    return model, tokenizer


def construct_inputs(
    tokenizer: PreTrainedTokenizer,
    model_config,
    batch_size: int,
    max_cache_len: int,
    past_key_values: Optional[StaticCache] = None,
    input_prompt: str = None,
    input_prompt_tokens: Optional[torch.Tensor] = None,
) -> dict:
    """
    Construct inputs including static cache.

    Args:
        tokenizer: Tokenizer instance
        model_config: Model configuration
        batch_size: Batch size
        max_cache_len: Maximum cache length
        input_prompt: Input text prompt (optional, defaults to DEFAULT_INPUT_PROMPT)
        input_prompt_tokens: Pre-tokenized input prompt (optional, overrides input_prompt)

    Returns:
        Dictionary containing input_ids, past_key_values, cache_position, and use_cache
    """
    if input_prompt_tokens is not None:
        if input_prompt_tokens.ndim != 1:
            raise ValueError(
                f"input_prompt_tokens must be 1D token IDs, got shape {tuple(input_prompt_tokens.shape)}"
            )
        if input_prompt_tokens.shape[0] > max_cache_len:
            input_prompt_tokens = input_prompt_tokens[:max_cache_len]

        input_ids = input_prompt_tokens.unsqueeze(0).expand(batch_size, -1).contiguous()
        inputs = {"input_ids": input_ids}
    else:
        if input_prompt is None:
            input_prompt = DEFAULT_INPUT_PROMPT
        input_prompt = [input_prompt] * batch_size

        # TODO: Only works on same length inputs for now
        prompt_len = len(input_prompt[0])
        assert all(
            len(prompt) == prompt_len for prompt in input_prompt
        ), "All input prompts must have the same length"

        inputs = tokenizer(
            input_prompt,
            return_tensors="pt",
            max_length=max_cache_len,
            truncation=True,
        )

    if past_key_values is None:
        # Static cache should be initialized on CPU and separately transferred to device
        # due to a trace/fusion issue. See https://github.com/tenstorrent/tt-xla/issues/1645
        static_cache = init_static_cache(
            config=model_config,
            batch_size=batch_size,
            max_cache_len=max_cache_len,
            device="cpu",
            dtype=torch.bfloat16,
        )
    else:
        static_cache = past_key_values
    input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
    cache_position: torch.Tensor = torch.arange(0, input_ids.shape[1])

    input_args = {
        "input_ids": input_ids,
        "past_key_values": static_cache,
        "cache_position": cache_position,
        "use_cache": True,
    }

    return input_args


def get_mesh(model_loader, mesh_config_fn):
    num_devices = xr.global_runtime_device_count()
    mesh_shape, mesh_name = mesh_config_fn(model_loader, num_devices)
    device_ids = np.array(range(num_devices))
    return Mesh(device_ids, mesh_shape, mesh_name)


def transfer_to_device(input_args: dict, device: torch.device) -> dict:
    """
    Transfer inputs to device.

    Args:
        input_args: Input arguments dictionary
        device: Target device

    Returns:
        input_args on device
    """
    for layer in input_args["past_key_values"].layers:
        layer.keys = layer.keys.to(device)
        layer.values = layer.values.to(device)
    input_args["input_ids"] = input_args["input_ids"].to(device)
    input_args["cache_position"] = input_args["cache_position"].to(device)
    return input_args


def check_transformers_version():
    """
    Check that transformers version is = 5.2.0.
    Raises RuntimeError if version is incompatible.
    """
    import packaging.version

    current_version = packaging.version.parse(transformers.__version__)
    max_version = packaging.version.parse("5.2.0")

    if current_version != max_version:
        raise RuntimeError(
            f"Transformers version {transformers.__version__} is not supported. "
            f"Please use version 5.2.0"
        )


def benchmark_llm_torch_xla(
    model_loader,
    model_variant,
    display_name,
    optimization_level,
    trace_enabled,
    batch_size,
    loop_count,
    task,
    data_format,
    input_sequence_length,
    experimental_weight_dtype,
    experimental_enable_permute_matmul_fusion,
    ttnn_perf_metrics_output_file,
    read_logits_fn,
    mesh_config_fn,
    shard_spec_fn,
    arch,
    required_pcc,
    fp32_dest_acc_en=None,
    accuracy_testing: bool = False,
    model_name_for_accuracy: str = None,
    hf_model_name_for_accuracy: str = None,
    max_output_tokens=None,
    decode_only: bool = False,
    weight_dtype_overrides: dict = None,
    input_output_sharding_spec=None,
    kv_cache_sharding_spec=None,
):
    """
    Benchmark an LLM (Large Language Model) using PyTorch and torch-xla.

    This function loads an LLM, compiles it with torch-xla for the Tenstorrent backend,
    and measures its text generation performance. It performs warmup runs, collects token
    generation metrics, and validates output correctness via PCC (Pearson Correlation Coefficient)
    or token accuracy (TOP1/TOP5) when accuracy_testing is enabled.
    The benchmark measures tokens per second on the device backends.

    Args:
        model_loader: Model loader instance for loading the LLM
        model_variant: Specific variant/version of the model to benchmark
        optimization_level: tt-mlir optimization level for compilation
        batch_size: Batch size for text generation
        loop_count: Number of inference iterations
        task: Task type
        data_format: Data precision format
        input_sequence_length: Length of input sequence for generation context
        experimental_weight_dtype: Weight dtype for block format conversion (e.g. "bfp_bf8", "bfp_bf4", or "" for none)
        experimental_enable_permute_matmul_fusion: Whether to enable permute matmul fusion optimization
        ttnn_perf_metrics_output_file: Path to save TTNN performance metrics
        read_logits_fn: Callback function to extract logits from model output
        required_pcc: Required PCC threshold for validation
        accuracy_testing: Whether to perform token accuracy testing
        model_name_for_accuracy: Model name for .refpt file lookup (required if accuracy_testing=True)
        hf_model_name_for_accuracy: Full HuggingFace model name for on-demand .refpt generation

    Returns:
        Benchmark result containing token generation performance metrics and model information
    """

    # Enforce bfloat16 only
    if data_format != "bfloat16":
        raise ValueError(
            f"Only bfloat16 data format is supported for llm benchmark. Got: {data_format}. "
            "Please use -df bfloat16"
        )

    if not model_loader:
        raise ValueError("Model loader must be specified for benchmark. ")

    if loop_count != 1:
        raise ValueError(
            f"Loop count must be 1 for llm benchmark (not yet supported). Got: {loop_count}. "
            "Please use -lp 1"
        )

    if not input_sequence_length or input_sequence_length <= 0:
        raise ValueError(
            f"Input sequence length must be a positive integer for llm benchmark. Got: {input_sequence_length}. "
            "Please use -isl <length> (e.g., -isl 128)"
        )

    if decode_only and accuracy_testing:
        raise ValueError("--decode-only cannot be combined with --accuracy-testing")

    if task != "text-generation":
        raise ValueError(
            f"Only 'text-generation' task is supported for llm benchmark. Got: {task}. "
            "Please use -t text-generation"
        )

    # Check transformers version
    check_transformers_version()

    xr.set_device_type("TT")

    # Set up for multi-chip if applicable
    if mesh_config_fn is not None and shard_spec_fn is not None:
        is_multichip = xr.global_runtime_device_count() > 1
        if is_multichip:
            os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
            xr.use_spmd()
    else:
        is_multichip = False

    # Set up config variables
    # WARNING: max_cache_len determines context window size for accuracy testing.
    # Reference outputs must be generated with total_length = max_cache_len for accurate comparison.
    # Changing this value requires regenerating ALL reference outputs (*.refpt files).
    max_cache_len: int = input_sequence_length

    # Connect the device
    device: torch.device = torch_xla.device()

    # Instantiate model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_loader, model_variant)
    full_model_name = model_loader.get_model_info(variant=model_variant).name

    # Initialize accuracy testing if enabled
    token_accuracy = None
    custom_input_prompt = None
    if accuracy_testing:
        token_accuracy, custom_input_prompt = init_accuracy_testing(
            model_name_for_accuracy=model_name_for_accuracy,
            max_cache_len=max_cache_len,
            tokenizer=tokenizer,
            hf_model_name=hf_model_name_for_accuracy,
        )

    # Construct inputs, including static cache
    input_args = construct_inputs(
        tokenizer,
        model.config,
        batch_size,
        max_cache_len,
        input_prompt=custom_input_prompt,
        input_prompt_tokens=(token_accuracy.input_prompt if accuracy_testing else None),
    )

    # Limit maximum generation count to fit within preallocated static cache
    if max_output_tokens is None:
        max_output_tokens = max_cache_len - input_args["input_ids"].shape[1]

    # Run CPU prefill (used as PCC baseline, or as decode-only prefill)
    if not accuracy_testing:
        cpu_wrapper = LLMSamplingWrapper(model, read_logits_fn, return_logits=True)
        cpu_wrapper.eval()
        cpu_output_logits, _ = generate_and_benchmark(
            cpu_wrapper,
            input_args,
            torch.device("cpu"),
            1,
            verbose=False,
            collect_logits=True,
        )

        if decode_only:
            # Save post-prefill state: CPU prefill populated the KV cache and
            # updated input_ids to the next token and cache_position to prompt_len.
            decode_only_input_ids = input_args["input_ids"].clone()
            decode_only_cache_position = input_args["cache_position"].clone()
            decode_only_cache = input_args["past_key_values"]

    # Transfer model and inputs to device
    input_args = construct_inputs(
        tokenizer,
        model.config,
        batch_size,
        max_cache_len,
        input_prompt=custom_input_prompt,
        input_prompt_tokens=(token_accuracy.input_prompt if accuracy_testing else None),
        past_key_values=decode_only_cache if decode_only else None,
    )
    input_args = transfer_to_device(input_args, device)
    model = model.to(device, dtype=torch.bfloat16)

    # Shard model if shard spec function is provided
    mesh = None
    if is_multichip:
        shard_specs = shard_spec_fn(model_loader, model)
        mesh = get_mesh(model_loader, mesh_config_fn)
        if shard_specs is not None:
            for tensor, shard_spec in shard_specs.items():
                xs.mark_sharding(tensor, mesh, shard_spec)

        # Also shard KV cache tensors created in input_args
        # This is hardcoded for all TP models, and should be moved to tt-forge-models.
        # https://github.com/tenstorrent/tt-xla/issues/4240
        kv_spec = kv_cache_sharding_spec or (None, "model", None, None)
        for layer in input_args["past_key_values"].layers:
            xs.mark_sharding(layer.keys, mesh, kv_spec)
            xs.mark_sharding(layer.values, mesh, kv_spec)

        # Shard input_ids
        if input_output_sharding_spec:
            xs.mark_sharding(input_args["input_ids"], mesh, input_output_sharding_spec)

        # Apply sharding constraint on lm_head output to all_gather logits
        if hasattr(model, "lm_head") and model.lm_head is not None:
            hook = sharding_constraint_hook(model.lm_head, mesh, (None, None, None))
            model.lm_head.register_forward_hook(hook)

    # Set XLA compilation options
    num_layers_override = getattr(model_loader, "num_layers", None)
    export_model_name = build_xla_export_name(
        model_name=display_name,
        num_layers=num_layers_override,
        batch_size=batch_size,
        input_sequence_length=input_sequence_length,
    )
    options = {
        "optimization_level": optimization_level,
        "enable_trace": trace_enabled,
        "export_path": MODULE_EXPORT_PATH,
        "export_model_name": export_model_name,
        "ttnn_perf_metrics_enabled": True,
        "ttnn_perf_metrics_output_file": ttnn_perf_metrics_output_file,
        "experimental_weight_dtype": experimental_weight_dtype,
        "experimental_enable_permute_matmul_fusion": experimental_enable_permute_matmul_fusion,
    }
    if fp32_dest_acc_en is not None:
        options["fp32_dest_acc_en"] = fp32_dest_acc_en

    torch_xla.set_custom_compile_options(options)

    # Apply per-tensor weight dtype overrides from explicit dict (takes priority).
    if weight_dtype_overrides:
        applied = apply_weight_dtype_overrides(model, weight_dtype_overrides)
        logger.info(f"Applied {len(applied)} weight dtype overrides from explicit dict")
    else:
        # Fall back to model's weight_dtype_configs JSON (auto-discovery).
        weight_dtype_config = model_loader.get_weight_dtype_config_path()
        if weight_dtype_config:
            applied = apply_weight_dtype_overrides(model, weight_dtype_config)
            logger.info(
                f"Applied {len(applied)} weight dtype overrides from {weight_dtype_config}"
            )
    # PERFORMANCE BENCHMARK
    # No logits returned to avoid OOM.
    perf_wrapper = LLMSamplingWrapper(
        model,
        read_logits_fn,
        return_logits=False,
        mesh=mesh,
        output_sharding_spec=input_output_sharding_spec,
    )
    perf_wrapper.eval()
    compiled_perf_model = torch.compile(perf_wrapper, backend="tt")

    # Warmup run (skip in decode-only mode)
    if not decode_only:
        print("Warming up...")
        warmup_tokens = min(MIN_STEPS, max_output_tokens)
        _, _ = generate_and_benchmark(
            compiled_perf_model,
            input_args,
            device,
            warmup_tokens,
            verbose=False,
            collect_logits=False,
        )

        tracy.signpost("warmup_complete")

    # Reconstruct inputs for the perf benchmark run
    input_args = construct_inputs(
        tokenizer,
        model.config,
        batch_size,
        max_cache_len,
        past_key_values=(
            input_args["past_key_values"] if not decode_only else decode_only_cache
        ),
        input_prompt=custom_input_prompt,
        input_prompt_tokens=(token_accuracy.input_prompt if accuracy_testing else None),
    )

    if decode_only:
        # Reset to post-prefill decode state (single token input)
        input_args["input_ids"] = decode_only_input_ids
        input_args["cache_position"] = decode_only_cache_position

    input_args = transfer_to_device(input_args, device)
    if input_output_sharding_spec:
        xs.mark_sharding(input_args["input_ids"], mesh, input_output_sharding_spec)

    # Run perf benchmark
    print(f"\nStarting performance benchmark...")
    ground_truth_for_benchmark = (
        token_accuracy.reference_tokens if accuracy_testing else None
    )
    _, iteration_times = generate_and_benchmark(
        compiled_perf_model,
        input_args,
        device,
        max_output_tokens,
        verbose=True,
        tokenizer=tokenizer,
        ground_truth_tokens=ground_truth_for_benchmark,
        collect_logits=False,
    )

    # ACCURACY BENCHMARK
    # Logits moved to CPU each step to avoid OOM.
    if not decode_only:
        accuracy_wrapper = LLMSamplingWrapper(
            model,
            read_logits_fn,
            return_logits=True,
            mesh=mesh,
            output_sharding_spec=input_output_sharding_spec,
        )
        accuracy_wrapper.eval()
        compiled_accuracy = torch.compile(accuracy_wrapper, backend="tt")

        accuracy_steps = max_output_tokens

        # Reconstruct inputs for accuracy run
        input_args = construct_inputs(
            tokenizer,
            model.config,
            batch_size,
            max_cache_len,
            past_key_values=input_args["past_key_values"],
            input_prompt=custom_input_prompt,
            input_prompt_tokens=(
                token_accuracy.input_prompt if accuracy_testing else None
            ),
        )
        input_args = transfer_to_device(input_args, device)
        if input_output_sharding_spec:
            xs.mark_sharding(input_args["input_ids"], mesh, input_output_sharding_spec)

        print(
            f"\nStarting accuracy benchmark "
            f"({accuracy_steps} step{'s' if accuracy_steps > 1 else ''})..."
        )
        output_logits, _ = generate_and_benchmark(
            compiled_accuracy,
            input_args,
            device,
            accuracy_steps,
            verbose=False,
            ground_truth_tokens=ground_truth_for_benchmark,
            collect_logits=True,
        )

    # Post-processing: derive predicted tokens for accuracy testing (all users)
    if accuracy_testing:
        batch_size_for_accuracy = output_logits[0].shape[0]
        per_user_predictions = []
        for user_idx in range(batch_size_for_accuracy):
            user_tokens = [
                logits.argmax(dim=-1)[user_idx].item() for logits in output_logits
            ]
            per_user_predictions.append(user_tokens)

    ttft_ns = iteration_times[0] if not decode_only else 0.0
    ttft_ms = ttft_ns / 1e6

    decode_iteration_times = iteration_times[1:]
    decode_total_time_ns = sum(decode_iteration_times)
    decode_total_time = decode_total_time_ns / 1e9

    # Calculate metrics (ignore first iteration for samples/sec)
    decode_total_tokens = len(decode_iteration_times)
    tokens_per_second = (
        (decode_total_tokens / decode_total_time) if decode_total_time > 0 else 0.0
    )

    metadata = get_benchmark_metadata()

    model_type = "text-generation"
    dataset_name = (
        "Tale of Two Cities (Reference Data)" if accuracy_testing else "Random Data"
    )

    # Extract number of layers from model config if available
    num_layers = (
        model.config.num_hidden_layers
        if hasattr(model.config, "num_hidden_layers")
        else -1
    )

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
        batch_size=batch_size,
        data_format=data_format,
        input_sequence_length=input_sequence_length,
        ttft_ms=ttft_ms,
    )

    evaluation_score = 0.0
    custom_measurements = [
        {
            "measurement_name": "ttft",
            "value": ttft_ms,
            "target": -1,
        },
    ]

    if accuracy_testing:
        # Compute token accuracy averaged across all users in the batch
        all_top1 = []
        all_top5 = []
        for user_idx, user_tokens in enumerate(per_user_predictions):
            user_top1, user_top5 = token_accuracy.compute_accuracy(user_tokens)
            all_top1.append(user_top1)
            all_top5.append(user_top5)

        top1_accuracy = sum(all_top1) / len(all_top1)
        top5_accuracy = sum(all_top5) / len(all_top5)
        print(
            "Token accuracy (averaged over {} users): TOP1={:.2f}%, TOP5={:.2f}%".format(
                len(all_top1), top1_accuracy * 100, top5_accuracy * 100
            )
        )

        # Store accuracy scores
        evaluation_score = top1_accuracy  # Use TOP1 as primary score
        custom_measurements.extend(
            [
                {
                    "measurement_name": "top1_accuracy",
                    "value": top1_accuracy * 100,  # Store as percentage
                },
                {
                    "measurement_name": "top5_accuracy",
                    "value": top5_accuracy * 100,  # Store as percentage
                },
            ]
        )
    elif decode_only:
        print("PCC verification skipped in decode-only mode")
    else:
        # Check PCC
        pcc_value = compute_pcc(
            output_logits[0][0], cpu_output_logits[0][0], required_pcc=required_pcc
        )
        print("PCC verification passed with PCC={:.6f}".format(pcc_value))

    # Get device count and mesh info for metrics
    device_count = xr.global_runtime_device_count()
    mesh_shape = tuple(mesh.shape()) if mesh is not None else None

    result = create_benchmark_result(
        full_model_name=full_model_name,
        model_type=model_type,
        dataset_name=dataset_name,
        num_layers=num_layers,
        batch_size=batch_size,
        input_size=(input_sequence_length,),
        loop_count=loop_count,
        data_format=data_format,
        total_time=decode_total_time,
        total_samples=decode_total_tokens,
        evaluation_score=evaluation_score,
        custom_measurements=custom_measurements,
        optimization_level=optimization_level,
        program_cache_enabled=True,
        trace_enabled=trace_enabled,
        experimental_weight_dtype=experimental_weight_dtype,
        model_info=full_model_name,
        display_name=display_name,
        torch_xla_enabled=True,
        backend="tt",
        device_name=socket.gethostname(),
        arch=arch or get_xla_device_arch(),
        input_is_image=False,
        input_sequence_length=input_sequence_length,
        device_count=device_count,
        mesh_shape=mesh_shape,
    )

    return result
