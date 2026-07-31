# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Multimodal / VLM benchmark harness (single forward pass, PCC vs CPU golden).

Modeled on ``vision_benchmark.py`` but with a different input contract:

* Inputs are a **dict of keyword arguments** (``input_ids`` / ``attention_mask`` /
  ``pixel_values`` / ``input_features`` / ...), not a single image tensor. The
  model is invoked as ``model(**inputs)``.
* Integer tensors (``input_ids``, ``attention_mask``, ``*_position_ids``,
  ``mm_token_type_ids``, ``input_features_mask``) **must stay integer** for
  embedding lookup / indexing. Only floating-point tensors (``pixel_values``,
  ``input_features``, ``pixel_values_videos``) are cast to ``data_format`` — see
  ``_move_inputs_to_device``. Force-casting integers to bf16 (as the vision
  harness does) would break the text/token path of a VLM.

The harness is tensor-parallel aware: when the loader exposes ``get_mesh_config``
/ ``load_shard_spec`` and more than one device is present, it enables SPMD, builds
the mesh and marks the weight shards — the same setup ``llm_benchmark.py`` uses.
This is required for large VLMs such as Gemma4-12B, which is sharded TP=8 over an
n300-llmbox (mesh ``(1, 8)``).
"""

# Built-in modules
import os
import socket
import time

import numpy as np
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh
from utils import (
    build_xla_export_name,
    compute_pcc,
    create_benchmark_result,
    get_benchmark_metadata,
    get_xla_device_arch,
    print_benchmark_results,
)

xr.set_device_type("TT")

# A single VLM forward pass is expensive (Gemma4-12B compiles for many minutes),
# so one warmup pass (which pays the compile cost) is enough before measuring.
WARMUP_STEPS = 1

MODULE_EXPORT_PATH = "modules"


def get_mesh(model_loader, mesh_config_fn):
    """Build the SPMD mesh from the loader's mesh-config callback.

    ``mesh_config_fn`` is the (unbound) ``ModelLoader.get_mesh_config`` method,
    so it is called as ``mesh_config_fn(model_loader, num_devices)`` — identical
    to ``llm_benchmark.get_mesh``.
    """
    num_devices = xr.global_runtime_device_count()
    mesh_shape, mesh_name = mesh_config_fn(model_loader, num_devices)
    device_ids = np.array(range(num_devices))
    return Mesh(device_ids, mesh_shape, mesh_name)


def _move_inputs_to_device(inputs, device, data_format):
    """Move a dict of kwargs to ``device``.

    Only floating-point tensors are cast to ``data_format``; integer tensors
    (``input_ids``, ``attention_mask``, ``*_position_ids``, ``mm_token_type_ids``,
    ``input_features_mask``) are left with their original dtype so embedding
    lookup and masking stay correct. ``None`` entries are dropped so the model
    stays on the intended modality path. Non-tensor values are passed through.
    """
    device_inputs = {}
    for key, value in inputs.items():
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            if torch.is_floating_point(value):
                value = value.to(dtype=data_format)
            device_inputs[key] = value.to(device)
        else:
            device_inputs[key] = value
    return device_inputs


def execute_and_measure(
    model, inputs_list, device, loop_count, extract_output_tensor_fn, data_format
):
    """Run ``loop_count`` forward passes and return (cpu_predictions, total_time_s)."""
    predictions = []
    start_time = time.perf_counter_ns()
    with torch.no_grad():
        outputs = []
        for i in range(loop_count):
            start_iteration_time = time.perf_counter_ns()
            # Move this batch's kwargs to device (dtype-preserving for ints).
            device_inputs = _move_inputs_to_device(inputs_list[i], device, data_format)

            # Model forward, non blocking.
            output = model(**device_inputs)

            # Extract the single tensor PCC is computed against (e.g. .logits).
            output = extract_output_tensor_fn(output)
            outputs.append(output)
            end_iteration_time = time.perf_counter_ns()
            print(
                f"Iteration {i} took {(end_iteration_time - start_iteration_time) / 1e6:.04} ms"
            )

        start_to_cpu_time = time.perf_counter_ns()
        predictions = [out.to("cpu") for out in outputs]
        end_to_cpu_time = time.perf_counter_ns()
        print(
            f"Moving all outputs to CPU took {(end_to_cpu_time - start_to_cpu_time) / 1e6:.04} ms"
        )

    end_time = time.perf_counter_ns()
    total_time = end_time - start_time
    print(f"Total time: {total_time / 1e9:.04}s for {loop_count} iterations")
    total_time /= 1e9
    return predictions, total_time


def benchmark_multimodal_torch_xla(
    model,
    model_loader,
    model_info_name,
    optimization_level,
    trace_enabled,
    batch_size,
    loop_count,
    data_format,
    ttnn_perf_metrics_output_file,
    load_inputs_fn,
    extract_output_tensor_fn,
    mesh_config_fn=None,
    shard_spec_fn=None,
    display_name=None,
    required_pcc=0.90,
    modality="image",
):
    """Benchmark a multimodal / VLM model using PyTorch and torch-xla.

    Compiles the model for the Tenstorrent backend, runs a single forward pass
    per iteration and validates the output against a CPU golden via PCC.

    Args:
        model: Loaded model instance in eval mode.
        model_loader: The loader instance (used for TP mesh / shard callbacks).
        model_info_name: Model name for identification and reporting.
        optimization_level: tt-mlir optimization level for compilation.
        trace_enabled: Whether to enable tracing.
        batch_size: Batch size for inference.
        loop_count: Number of inference iterations to benchmark.
        data_format: torch.dtype for model precision (bf16 / fp32).
        ttnn_perf_metrics_output_file: Path to save TTNN performance metrics.
        load_inputs_fn: fn(dtype) -> dict of model kwargs for one batch.
        extract_output_tensor_fn: fn(output) -> single tensor for PCC.
        mesh_config_fn: Optional (unbound) ``get_mesh_config`` for TP.
        shard_spec_fn: Optional (unbound) ``load_shard_spec`` for TP.
        required_pcc: Minimum PCC threshold for output validation.
        modality: One of "text" / "image" / "audio" / "video" (reporting only).

    Returns:
        Benchmark result dict containing performance metrics and model info.
    """
    xr.set_device_type("TT")

    # Enable SPMD for multi-chip tensor-parallel runs (mirrors llm_benchmark).
    is_multichip = False
    if mesh_config_fn is not None and shard_spec_fn is not None:
        is_multichip = xr.global_runtime_device_count() > 1
        if is_multichip:
            os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
            xr.use_spmd()

    framework_model = model

    # Generate inputs (CPU, natural dtypes).
    inputs = [load_inputs_fn(data_format) for _ in range(loop_count)]

    # Generate the golden output for PCC on CPU, using the same dtype policy the
    # device path uses (floats -> data_format, ints preserved).
    golden_cpu_inputs = _move_inputs_to_device(
        inputs[0], torch.device("cpu"), data_format
    )
    with torch.no_grad():
        golden_output = framework_model(**golden_cpu_inputs)
        golden_output = extract_output_tensor_fn(golden_output).to("cpu")

    export_model_name = build_xla_export_name(
        model_name=display_name,
        num_layers=getattr(model_loader, "num_layers", None),
        batch_size=batch_size,
        input_sequence_length=None,
    )

    options = {
        "optimization_level": optimization_level,
        "export_path": MODULE_EXPORT_PATH,
        "export_model_name": export_model_name,
        "ttnn_perf_metrics_enabled": True,
        "ttnn_perf_metrics_output_file": ttnn_perf_metrics_output_file,
        "enable_trace": trace_enabled,
    }
    torch_xla.set_custom_compile_options(options)

    device = torch_xla.device()

    # Transfer model to device, then shard (mark_sharding needs device tensors).
    framework_model = framework_model.to(device, dtype=data_format)

    mesh = None
    if is_multichip:
        shard_specs = shard_spec_fn(model_loader, framework_model)
        mesh = get_mesh(model_loader, mesh_config_fn)
        if shard_specs is not None:
            for tensor, shard_spec in shard_specs.items():
                xs.mark_sharding(tensor, mesh, shard_spec)

    framework_model.compile(backend="tt")

    # Warmup (pays the compile cost).
    print("Starting warmup...")
    warmup_loop_count = min(WARMUP_STEPS, loop_count)
    execute_and_measure(
        model=framework_model,
        inputs_list=inputs[:warmup_loop_count],
        device=device,
        loop_count=warmup_loop_count,
        extract_output_tensor_fn=extract_output_tensor_fn,
        data_format=data_format,
    )
    print("Warmup completed.")

    # Benchmark.
    print("Starting benchmark...")
    predictions, total_time = execute_and_measure(
        model=framework_model,
        inputs_list=inputs,
        device=device,
        loop_count=loop_count,
        extract_output_tensor_fn=extract_output_tensor_fn,
        data_format=data_format,
    )
    print("Benchmark completed.")

    total_samples = batch_size * loop_count
    samples_per_sec = total_samples / total_time

    metadata = get_benchmark_metadata()

    full_model_name = model_info_name
    model_type = f"Multimodal ({modality}), Random Input Data"
    dataset_name = "Random Data"
    num_layers = getattr(model_loader, "num_layers", None) or -1

    if data_format == torch.bfloat16:
        data_format_str = "bfloat16"
    elif data_format == torch.float32:
        data_format_str = "float32"
    else:
        raise ValueError(f"Unsupported data format: {data_format}")

    # Multimodal inputs are a heterogeneous dict, not a single (C,H,W) tensor.
    input_size = (-1, -1, -1)

    print_benchmark_results(
        model_title=full_model_name,
        full_model_name=full_model_name,
        model_type=model_type,
        dataset_name=dataset_name,
        date=metadata["date"],
        machine_name=metadata["machine_name"],
        total_time=total_time,
        total_samples=total_samples,
        samples_per_sec=samples_per_sec,
        evaluation_score=0.0,
        batch_size=batch_size,
        data_format=data_format_str,
        input_size=input_size,
    )

    # Evaluate PCC.
    pcc_value = compute_pcc(predictions[0], golden_output)
    assert (
        pcc_value >= required_pcc
    ), f"PCC comparison failed. PCC={pcc_value:.6f}, Required={required_pcc}"
    print(f"PCC verification passed with PCC={pcc_value:.6f}")

    device_count = xr.global_runtime_device_count()
    mesh_shape = tuple(mesh.shape()) if mesh is not None else None

    result = create_benchmark_result(
        full_model_name=full_model_name,
        model_type=model_type,
        dataset_name=dataset_name,
        num_layers=num_layers,
        batch_size=batch_size,
        input_size=input_size,
        loop_count=loop_count,
        data_format=data_format_str,
        total_time=total_time,
        total_samples=total_samples,
        evaluation_score=0.0,
        optimization_level=optimization_level,
        program_cache_enabled=True,
        trace_enabled=trace_enabled,
        model_info=model_info_name,
        display_name=display_name,
        torch_xla_enabled=True,
        backend="tt",
        device_name=socket.gethostname(),
        arch=get_xla_device_arch(),
        device_count=device_count,
        mesh_shape=mesh_shape,
        input_is_image=False,
    )

    return result
