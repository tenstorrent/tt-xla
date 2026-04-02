# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Built-in modules
import os
import socket
import sys
from typing import Callable, Optional

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

# Steady-state tokens/s uses only the last N timed forward steps (rest treated as warmup).
THROUGHPUT_STEADY_STATE_LAST_N = 100

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
    # Debug: verify num_layers override actually applied to the loaded model config.
    if hasattr(model_loader, "num_layers"):
        print(f"[num_layers] model_loader.num_layers={getattr(model_loader, 'num_layers')}")
    if hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
        print(f"[num_layers] model.config.num_hidden_layers={model.config.num_hidden_layers}")
    if hasattr(model.config, "layer_types"):
        # Some models (e.g. GPT-OSS) use layer_types length as the authoritative layer count
        # for KV cache sizing. Keep it consistent with num_hidden_layers when overridden.
        n = (
            int(model.config.num_hidden_layers)
            if hasattr(model.config, "num_hidden_layers") and model.config.num_hidden_layers is not None
            else len(model.config.layer_types)
        )
        model.config.layer_types = ["full_attention"] * n
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
        if hasattr(model_config, "num_hidden_layers"):
            print(f"[num_layers] StaticCache init: config.num_hidden_layers={model_config.num_hidden_layers}")
        if hasattr(static_cache, "layers"):
            print(f"[num_layers] StaticCache layers={len(static_cache.layers)}")
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
    Check that transformers version is <= 4.57.1.
    Raises RuntimeError if version is incompatible.
    """
    import packaging.version

    current_version = packaging.version.parse(transformers.__version__)
    max_version = packaging.version.parse("4.57.1")

    if current_version > max_version:
        raise RuntimeError(
            f"Transformers version {transformers.__version__} is not supported. "
            f"Please use version <= 4.57.1"
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
    input_sharding_fn: Optional[Callable] = None,
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
        input_sharding_fn: Optional ``(mesh, input_args) -> None`` to mark activation shardings
            (e.g. batch-parallel ``input_ids``). When set, KV cache tensors are batch-sharded too.

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

    # Get CPU result (skip in accuracy testing mode - not needed with ground truth)
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
        cpu_logits = cpu_output_logits[0]

    # Transfer model and inputs to device
    input_args = construct_inputs(
        tokenizer,
        model.config,
        batch_size,
        max_cache_len,
        input_prompt=custom_input_prompt,
        input_prompt_tokens=(token_accuracy.input_prompt if accuracy_testing else None),
    )
    input_args = transfer_to_device(input_args, device)
    model = model.to(device, dtype=torch.bfloat16)

    # Shard model if shard spec function is provided
    mesh = None
    # Fresh prepare_step_inputs_fn per decode loop so DP sharding skips prefill (step 0) each time.
    make_prepare_step_inputs_fn = lambda: None
    if is_multichip:
        shard_specs = shard_spec_fn(model_loader, model)
        mesh = get_mesh(model_loader, mesh_config_fn)
        if shard_specs is not None:
            for tensor, shard_spec in shard_specs.items():
                xs.mark_sharding(tensor, mesh, shard_spec)

        # Decode-only DP: keep prefill unsharded and apply activation batch sharding
        # inside the decode loop (after step 0) via prepare_step_inputs_fn.
        if input_sharding_fn is not None:
            def _make_prepare_step_inputs_fn(_mesh=mesh, _fn=input_sharding_fn):
                _dp_step = {"i": 0}

                def _prepare_step_inputs_fn(ia):
                    if _dp_step["i"] > 0:
                        _fn(_mesh, ia)
                    _dp_step["i"] += 1

                return _prepare_step_inputs_fn

            make_prepare_step_inputs_fn = _make_prepare_step_inputs_fn

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

    ground_truth_for_benchmark = (
        token_accuracy.reference_tokens if accuracy_testing else None
    )

    # Apply per-tensor weight dtype overrides from model's weight_dtype_configs JSON.
    weight_dtype_config = model_loader.get_weight_dtype_config_path()
    if weight_dtype_config:
        applied = apply_weight_dtype_overrides(model, weight_dtype_config)
        logger.info(
            f"Applied {len(applied)} weight dtype overrides from {weight_dtype_config}"
        )
    # PERFORMANCE BENCHMARK
    # No logits returned to avoid OOM. SPMD input sharding uses prepare_step_inputs_fn.
    perf_wrapper = LLMSamplingWrapper(model, read_logits_fn, return_logits=False)
    perf_wrapper.eval()
    compiled_perf_model = torch.compile(perf_wrapper, backend="tt")

    # Warmup run
    print("Warming up...")
    warmup_tokens = min(MIN_STEPS, max_output_tokens)
    _, _ = generate_and_benchmark(
        compiled_perf_model,
        input_args,
        device,
        warmup_tokens,
        verbose=False,
        collect_logits=False,
        prepare_step_inputs_fn=make_prepare_step_inputs_fn(),
        ground_truth_tokens=ground_truth_for_benchmark,
    )

    tracy.signpost("warmup_complete")

    # DP: recompile against a fresh KV cache before the timed perf run.
    if is_multichip and input_sharding_fn is not None:
        torch._dynamo.reset()
        compile_warmup_wrapper = LLMSamplingWrapper(
            model, read_logits_fn, return_logits=False
        )
        compile_warmup_wrapper.eval()
        compiled_compile_warmup = torch.compile(compile_warmup_wrapper, backend="tt")

        compile_warmup_args = construct_inputs(
            tokenizer,
            model.config,
            batch_size,
            max_cache_len,
            input_prompt=custom_input_prompt,
            input_prompt_tokens=(
                token_accuracy.input_prompt if accuracy_testing else None
            ),
        )
        compile_warmup_args = transfer_to_device(compile_warmup_args, device)

        _, _ = generate_and_benchmark(
            compiled_compile_warmup,
            compile_warmup_args,
            device,
            min(2, max_output_tokens),
            verbose=False,
            collect_logits=False,
            prepare_step_inputs_fn=make_prepare_step_inputs_fn(),
            ground_truth_tokens=ground_truth_for_benchmark,
        )

    # Reconstruct inputs for the perf benchmark run.
    input_args = construct_inputs(
        tokenizer,
        model.config,
        batch_size,
        max_cache_len,
        past_key_values=(
            None if input_sharding_fn is not None else input_args["past_key_values"]
        ),
        input_prompt=custom_input_prompt,
        input_prompt_tokens=(token_accuracy.input_prompt if accuracy_testing else None),
    )
    input_args = transfer_to_device(input_args, device)

    # Run perf benchmark
    print(f"\nStarting performance benchmark...")
    _, iteration_times = generate_and_benchmark(
        compiled_perf_model,
        input_args,
        device,
        max_output_tokens,
        verbose=True,
        tokenizer=tokenizer,
        ground_truth_tokens=ground_truth_for_benchmark,
        collect_logits=False,
        prepare_step_inputs_fn=make_prepare_step_inputs_fn(),
    )

    # ACCURACY BENCHMARK
    # Logits moved to CPU each step to avoid OOM.
    accuracy_wrapper = LLMSamplingWrapper(model, read_logits_fn, return_logits=True)
    accuracy_wrapper.eval()
    torch._dynamo.reset()
    compiled_accuracy = torch.compile(accuracy_wrapper, backend="tt")

    accuracy_steps = max_output_tokens

    # Fresh KV cache: perf run mutates cache in-place; reusing it breaks logits.
    input_args = construct_inputs(
        tokenizer,
        model.config,
        batch_size,
        max_cache_len,
        past_key_values=None,
        input_prompt=custom_input_prompt,
        input_prompt_tokens=(token_accuracy.input_prompt if accuracy_testing else None),
    )
    input_args = transfer_to_device(input_args, device)

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
        prepare_step_inputs_fn=make_prepare_step_inputs_fn(),
    )

    # Post-processing: derive predicted tokens for accuracy testing
    if accuracy_testing:
        predicted_tokens = [
            logits[:, -1].argmax(dim=-1)[0].item() for logits in output_logits
        ]

    ttft_ns = iteration_times[0]
    ttft_ms = ttft_ns / 1e6

    # Throughput: last N of all timed steps (e.g. last 100 of 110), shorter runs use all steps.
    n_tail = THROUGHPUT_STEADY_STATE_LAST_N
    throughput_times = (
        iteration_times[-n_tail:]
        if len(iteration_times) >= n_tail
        else iteration_times
    )
    throughput_time_ns = sum(throughput_times)
    throughput_time_s = throughput_time_ns / 1e9
    throughput_steps = len(throughput_times)
    tokens_per_second = (
        (throughput_steps / throughput_time_s) if throughput_time_s > 0 else 0.0
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
        total_time=throughput_time_s,
        total_samples=throughput_steps,
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
        # Compute token accuracy from predictions (after generation completes)
        top1_accuracy, top5_accuracy = token_accuracy.compute_accuracy(predicted_tokens)
        print(
            "Token accuracy: TOP1={:.2f}%, TOP5={:.2f}%".format(
                top1_accuracy * 100, top5_accuracy * 100
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
    else:
        # PCC: 1st arg = compiled TT logits slice; 2nd = CPU reference (compute_pcc param names are legacy).
        tt_slice = output_logits[0][0]
        cpu_slice = cpu_logits[0]
        print(
            "[llm_benchmark PCC] Same batch row / step index in output_logits vs cpu_logits.\n"
            f"  TT logits row:   shape={tuple(tt_slice.shape)} dtype={tt_slice.dtype} device={tt_slice.device}\n"
            f"  CPU logits row: shape={tuple(cpu_slice.shape)} dtype={cpu_slice.dtype} device={cpu_slice.device}"
        )
        pcc_value = compute_pcc(tt_slice, cpu_slice, required_pcc=required_pcc)
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
        total_time=throughput_time_s,
        total_samples=throughput_steps,
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
