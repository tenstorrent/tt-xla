# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import socket
import time
from typing import Callable, List, Tuple

import jax
import numpy as np
import torch
from jax import device_put
from transformers import FlaxResNetForImageClassification
from tt_jax import serialize_compiled_artifacts_to_disk
from accuracy import compute_pcc
from naming import perf_metrics_filename, sanitize_name
from reporting import (
    create_benchmark_result,
    get_benchmark_metadata,
    print_benchmark_results,
    write_benchmark_json,
)
from runtime import MODULE_EXPORT_PATH, get_jax_device_arch

WARMUP_STEPS = 8


def execute_and_measure_fps(
    compiled_model: Callable,
    inputs: List["jax.Array"],
    params: dict,
    loop_count: int,
) -> Tuple[List["jax.Array"], float]:
    """Time ``loop_count`` batches; return (predictions on CPU, total_time_seconds)."""
    predictions = []
    start_time = time.perf_counter_ns()

    outputs = []
    for i in range(loop_count):
        start_iteration_time = time.perf_counter_ns()
        output = compiled_model(inputs[i], train=False, params=params)
        outputs.append(output.logits)
        end_iteration_time = time.perf_counter_ns()
        print(
            f"Iteration {i} took {(end_iteration_time - start_iteration_time) / 1e6:.04} ms"
        )

    start_to_cpu_time = time.perf_counter_ns()
    predictions = [jax.device_put(out, jax.devices("cpu")[0]) for out in outputs]
    end_to_cpu_time = time.perf_counter_ns()
    print(
        f"Moving all outputs to CPU took {(end_to_cpu_time - start_to_cpu_time) / 1e6:.04} ms"
    )

    total_time = (time.perf_counter_ns() - start_time) / 1e9  # ns -> s
    print(f"Total time: {total_time:.04}s for {loop_count} iterations")
    return predictions, total_time


def benchmark_resnet_jax(
    variant: str,
    input_size: Tuple[int, int, int],
    batch_size: int,
    loop_count: int,
    data_format: str,
    model_name: str,
    optimization_level: int = 1,
    program_cache_enabled: bool = False,
    trace_enabled: bool = False,
    experimental_weight_dtype: str = "",
    required_pcc: float = 0.97,
) -> dict:
    """Benchmark a ResNet model with JAX / tt-jax on the Tenstorrent backend.

    Compiles the model, times inference, and validates output via PCC.
    ``variant`` is a HuggingFace id (e.g. ``"microsoft/resnet-50"``).
    ``input_size`` is channel-first (channels, height, width).

    Returns the standardized result dict.
    """
    tt_device = jax.devices("tt")[0]
    cpu_device = jax.devices("cpu")[0]

    with jax.default_device(cpu_device):
        # Instantiate the model on CPU
        framework_model = FlaxResNetForImageClassification.from_pretrained(
            variant,
            from_pt=True,
        )
        model_info = variant

    # Generate inputs (on CPU - RNG requires an unsupported SHLO op)
    inputs = []
    for i in range(loop_count):
        with jax.default_device(cpu_device):
            input_sample = jax.random.normal(
                jax.random.PRNGKey(i),
                (batch_size, input_size[0], input_size[1], input_size[2]),
            )
        inputs.append(input_sample)

    # Generate golden output for PCC calculation (run on CPU)
    with jax.default_device(cpu_device):
        golden_input = inputs[0]
        golden_output = framework_model(
            golden_input, train=False, params=framework_model.params
        )
        golden_output = golden_output.logits

    # Move inputs to TT device
    inputs = [device_put(inp, tt_device) for inp in inputs]

    # Move model parameters to TT device
    framework_model.params = jax.tree_util.tree_map(
        lambda x: device_put(x, tt_device), framework_model.params
    )

    # Serialize compiled artifacts
    serialize_compiled_artifacts_to_disk(
        framework_model,
        inputs[0],
        output_prefix=f"{MODULE_EXPORT_PATH}/{model_name}",
        params=framework_model.params,
    )

    # Compile the forward pass
    compiled_fwd = jax.jit(framework_model.__call__, static_argnames=["train"])

    print("Starting warmup...")
    warmup_loop_count = min(WARMUP_STEPS, loop_count)
    warmup_inputs = inputs[:warmup_loop_count]
    execute_and_measure_fps(
        compiled_model=compiled_fwd,
        inputs=warmup_inputs,
        params=framework_model.params,
        loop_count=warmup_loop_count,
    )
    print("Warmup completed.")

    print("Starting benchmark...")
    predictions, total_time = execute_and_measure_fps(
        compiled_model=compiled_fwd,
        inputs=inputs,
        params=framework_model.params,
        loop_count=loop_count,
    )
    print("Benchmark completed.")

    total_samples = batch_size * loop_count
    samples_per_sec = total_samples / total_time

    metadata = get_benchmark_metadata()

    full_model_name = "Resnet 50 HF"
    model_type = "Classification, Random Input Data"
    dataset_name = full_model_name + ", Random Data"
    num_layers = 50

    print_benchmark_results(
        model_title="Resnet",
        full_model_name=full_model_name,
        model_type=model_type,
        dataset_name=dataset_name,
        date=metadata["date"],
        machine_name=metadata["machine_name"],
        total_time=total_time,
        total_samples=total_samples,
        samples_per_sec=samples_per_sec,
        batch_size=batch_size,
        data_format=data_format,
        input_size=input_size,
    )

    # Convert JAX arrays to PyTorch tensors for PCC comparison.
    golden_tensor = torch.from_numpy(np.asarray(golden_output))
    prediction_tensor = torch.from_numpy(np.asarray(predictions[0]))
    pcc_value = compute_pcc(golden_tensor, prediction_tensor)
    assert (
        pcc_value >= required_pcc
    ), f"PCC comparison failed. PCC={pcc_value:.6f}, Required={required_pcc}"
    print(f"PCC verification passed with PCC={pcc_value:.6f}")

    result = create_benchmark_result(
        full_model_name=full_model_name,
        model_type=model_type,
        dataset_name=dataset_name,
        num_layers=num_layers,
        batch_size=batch_size,
        input_size=input_size,
        loop_count=loop_count,
        data_format=data_format,
        total_time=total_time,
        total_samples=total_samples,
        optimization_level=optimization_level,
        program_cache_enabled=program_cache_enabled,
        trace_enabled=trace_enabled,
        experimental_weight_dtype=experimental_weight_dtype,
        model_info=model_info,
        torch_xla_enabled=False,
        device_name=socket.gethostname(),
        arch=get_jax_device_arch(),
    )

    return result


def test_resnet_jax(output_file):
    # Configuration
    variant = "microsoft/resnet-50"
    batch_size = 8
    loop_count = 32
    input_size = (3, 224, 224)
    data_format = "float32"
    model_name = "resnet_jax"

    sanitized_model_name = sanitize_name(model_name)
    ttnn_perf_metrics_output_file = perf_metrics_filename(sanitized_model_name)

    print(f"Running JAX benchmark for model: {model_name}")
    print(f"""Configuration:
    variant={variant}
    batch_size={batch_size}
    loop_count={loop_count}
    input_size={input_size}
    data_format={data_format}
    ttnn_perf_metrics_output_file={ttnn_perf_metrics_output_file}
    """)

    results = benchmark_resnet_jax(
        variant=variant,
        input_size=input_size,
        batch_size=batch_size,
        loop_count=loop_count,
        data_format=data_format,
        model_name=model_name,
    )

    if output_file:
        write_benchmark_json(
            results,
            output_file,
            model_rawname=model_name,
            ttnn_perf_metrics_file=ttnn_perf_metrics_output_file,
        )
