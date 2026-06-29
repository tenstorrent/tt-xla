# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import time
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch_xla
from accuracy import assert_pcc
from naming import build_xla_export_name
from reporting import (
    create_benchmark_result,
    get_benchmark_metadata,
    print_benchmark_results,
)
from runtime import init_tt_runtime, set_compile_options, tt_xla_device_fields

init_tt_runtime()

WARMUP_STEPS = 32


def execute_and_measure_fps(
    model: Callable,
    inputs: List[torch.Tensor],
    device: torch.device,
    loop_count: int,
    extract_output_tensor_fn: Callable,
) -> Tuple[List[torch.Tensor], float]:
    """Time ``loop_count`` batches; return (predictions on CPU, total_time_seconds).

    ``extract_output_tensor_fn`` pulls the tensor out of the model output
    (e.g. ``.logits`` from a HF output).
    Outputs are kept on device through the timed loop and moved to CPU together
    at the end to avoid per-step syncs.
    """
    predictions = []
    start_time = time.perf_counter_ns()
    with torch.no_grad():
        outputs = []
        for i in range(loop_count):
            start_iteration_time = time.perf_counter_ns()
            device_input = inputs[i].to(device)
            output = model(device_input)
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

    total_time = (time.perf_counter_ns() - start_time) / 1e9
    print(f"Total time: {total_time:.04}s for {loop_count} iterations")
    return predictions, total_time


def benchmark_vision_torch_xla(
    model: torch.nn.Module,
    model_info_name: str,
    optimization_level: int,
    trace_enabled: bool,
    batch_size: int,
    loop_count: int,
    input_size: Tuple[int, int, int],
    data_format: torch.dtype,
    ttnn_perf_metrics_output_file: str,
    load_inputs_fn: Callable,
    extract_output_tensor_fn: Callable,
    display_name: Optional[str] = None,
    required_pcc: float = 0.97,
) -> dict:
    """Benchmark a vision model with torch-xla on the Tenstorrent backend.

    Compiles the model, warms up, times inference, and validates output via PCC.
    ``input_size`` is channel-first (channels, height, width).

    Per-model hooks:
    - ``load_inputs_fn(batch_size, dtype) -> batch``
    - ``extract_output_tensor_fn(output) -> tensor`` (e.g. ``.logits``)

    Returns the standardized benchmark-result dict.
    """
    framework_model = model

    inputs = [load_inputs_fn(batch_size, data_format) for _ in range(loop_count)]

    # CPU golden for the PCC check.
    golden_input = inputs[0]
    with torch.no_grad():
        golden_output = framework_model(golden_input)
        golden_output = extract_output_tensor_fn(golden_output)

    export_model_name = build_xla_export_name(
        model_name=display_name,
        num_layers=None,
        batch_size=batch_size,
        input_sequence_length=None,
    )

    set_compile_options(
        optimization_level=optimization_level,
        export_model_name=export_model_name,
        ttnn_perf_metrics_output_file=ttnn_perf_metrics_output_file,
        enable_trace=trace_enabled,
    )

    framework_model.compile(backend="tt")
    device = torch_xla.device()

    # Clear num_batches_tracked on BatchNorm layers to avoid creating an extra
    # XLA graph for these unused buffers. In eval mode, num_batches_tracked is
    # never used, but if left as a tensor it gets transferred to the XLA device
    # and creates a separate constant sync graph.
    for m in framework_model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.num_batches_tracked = None

    framework_model = framework_model.to(device, dtype=data_format)

    print("Starting warmup...")
    warmup_loop_count = min(WARMUP_STEPS, loop_count)
    warmup_inputs = inputs[:warmup_loop_count]
    execute_and_measure_fps(
        model=framework_model,
        inputs=warmup_inputs,
        device=device,
        loop_count=warmup_loop_count,
        extract_output_tensor_fn=extract_output_tensor_fn,
    )
    print("Warmup completed.")

    print("Starting benchmark...")
    predictions, total_time = execute_and_measure_fps(
        model=framework_model,
        inputs=inputs,
        device=device,
        loop_count=loop_count,
        extract_output_tensor_fn=extract_output_tensor_fn,
    )
    print("Benchmark completed.")

    total_samples = batch_size * loop_count
    samples_per_sec = total_samples / total_time

    metadata = get_benchmark_metadata()

    full_model_name = model_info_name
    model_type = "Vision, Random Input Data"
    dataset_name = "Random Data"
    num_layers = -1

    if data_format == torch.bfloat16:
        data_format_str = "bfloat16"
    elif data_format == torch.float32:
        data_format_str = "float32"
    else:
        raise ValueError(f"Unsupported data format: {data_format}")

    evaluation_score = 0.0
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
        evaluation_score=evaluation_score,
        batch_size=batch_size,
        data_format=data_format_str,
        input_size=input_size,
    )

    assert_pcc(predictions[0], golden_output, required_pcc)

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
        evaluation_score=evaluation_score,
        optimization_level=optimization_level,
        trace_enabled=trace_enabled,
        model_info=model_info_name,
        display_name=display_name,
        **tt_xla_device_fields(),
    )

    return result
