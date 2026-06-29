# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import time
from typing import Callable, List, Optional, Tuple

import torch
import torch_xla
from accuracy import assert_pcc
from model_utils import move_to_cpu
from naming import build_xla_export_name
from reporting import (
    create_benchmark_result,
    get_benchmark_metadata,
    print_benchmark_results,
)
from runtime import init_tt_runtime, set_compile_options, tt_xla_device_fields

init_tt_runtime()

WARMUP_STEPS = 3  # Number of warmup iterations before benchmarking


def run_encoder_model(
    model: Callable,
    raw_inputs: List[str],
    preprocess_fn: Callable,
    device: torch.device,
    output_processor_fn: Callable,
) -> torch.Tensor:
    """Tokenize, run, and post-process sentences into [B, H] embeddings.

    Per-model hooks:
    - ``preprocess_fn(sentences, device) -> input kwargs``
    - ``output_processor_fn(outputs, model_inputs) -> embeddings``
    """
    model_inputs = preprocess_fn(raw_inputs, device)
    outputs = model(**model_inputs)

    # Move all outputs to CPU before running processor_fn to avoid
    # creating extra XLA graphs for post-processing operations
    outputs = move_to_cpu(outputs)
    model_inputs = move_to_cpu(model_inputs)

    return output_processor_fn(outputs, model_inputs)


def warmup_encoder_model(
    model: Callable,
    raw_inputs: List[str],
    preprocess_fn: Callable,
    device: torch.device,
    output_processor_fn: Callable,
    loop_count: int,
) -> None:
    """Run ``loop_count`` warmup iterations of the encoder model."""
    print("Warming up the device...")

    with torch.no_grad():
        for i in range(loop_count):
            run_encoder_model(
                model, raw_inputs, preprocess_fn, device, output_processor_fn
            )

    print("Warming up completed.")


def measure_fps_encoder_model(
    model: Callable,
    raw_inputs: List[str],
    preprocess_fn: Callable,
    device: torch.device,
    output_processor_fn: Callable,
    loop_count: int,
) -> Tuple[List[torch.Tensor], float]:
    """Time ``loop_count`` iterations; return (predictions, total_time_seconds)."""
    print("Starting benchmark loop...")

    predictions = []
    iteration_times = []

    with torch.no_grad():
        for i in range(loop_count):
            start_time = time.perf_counter_ns()
            output = run_encoder_model(
                model, raw_inputs, preprocess_fn, device, output_processor_fn
            )
            predictions.append(output)
            end_time = time.perf_counter_ns()

            iteration_times.append(end_time - start_time)
            print(
                f"Iteration\t{i+1}/{loop_count}\ttook {iteration_times[-1] / 1e6:.04} ms"
            )

    total_time = sum(iteration_times) / 1e9  # ns -> s
    return predictions, total_time


def benchmark_encoder_torch_xla(
    model: torch.nn.Module,
    model_info_name: str,
    optimization_level: int,
    trace_enabled: bool,
    batch_size: int,
    input_sequence_length: int,
    loop_count: int,
    data_format: str,
    ttnn_perf_metrics_output_file: str,
    load_inputs_fn: Callable,
    preprocess_fn: Callable,
    output_processor_fn: Callable,
    display_name: Optional[str] = None,
    num_layers_override: Optional[int] = None,
    required_pcc: float = 0.97,
    experimental_weight_dtype: str = "",
    experimental_enable_permute_matmul_fusion: bool = False,
) -> dict:
    """Benchmark an encoder model with torch-xla on the Tenstorrent backend.

    Compiles the model, warms up, times end-to-end inference (tokenization runs
    per iteration and is included in the timing), and validates output via PCC.

    Per-model hooks:
    - ``load_inputs_fn(batch_size) -> sentences``
    - ``preprocess_fn(sentences, device) -> input kwargs``
    - ``output_processor_fn(outputs, model_inputs) -> embeddings``

    Returns the standardized benchmark-result dict.
    """
    framework_model = model

    raw_inputs = load_inputs_fn(batch_size)

    print("Generating golden output on CPU...")
    with torch.no_grad():
        golden_output = run_encoder_model(
            framework_model, raw_inputs, preprocess_fn, "cpu", output_processor_fn
        )

    export_model_name = build_xla_export_name(
        model_name=display_name,
        num_layers=num_layers_override,
        batch_size=batch_size,
        input_sequence_length=input_sequence_length,
    )

    set_compile_options(
        optimization_level=optimization_level,
        export_model_name=export_model_name,
        ttnn_perf_metrics_output_file=ttnn_perf_metrics_output_file,
        enable_trace=trace_enabled,
        experimental_weight_dtype=experimental_weight_dtype,
        experimental_enable_permute_matmul_fusion=experimental_enable_permute_matmul_fusion,
    )

    framework_model.compile(backend="tt")
    device = torch_xla.device()
    framework_model = framework_model.to(device)

    warmup_count = min(WARMUP_STEPS, loop_count)
    warmup_encoder_model(
        model=framework_model,
        raw_inputs=raw_inputs,
        preprocess_fn=preprocess_fn,
        device=device,
        output_processor_fn=output_processor_fn,
        loop_count=warmup_count,
    )

    predictions, total_time = measure_fps_encoder_model(
        model=framework_model,
        raw_inputs=raw_inputs,
        preprocess_fn=preprocess_fn,
        device=device,
        output_processor_fn=output_processor_fn,
        loop_count=loop_count,
    )

    evaluation_score = assert_pcc(predictions[0], golden_output, required_pcc)

    total_samples = batch_size * loop_count
    samples_per_sec = total_samples / total_time

    metadata = get_benchmark_metadata()

    full_model_name = model_info_name
    model_type = "Encoder, Text Embedding"
    dataset_name = "Benchmark Sentences"
    num_layers = -1

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
        data_format=data_format,
        input_sequence_length=input_sequence_length,
    )

    result = create_benchmark_result(
        full_model_name=full_model_name,
        model_type=model_type,
        dataset_name=dataset_name,
        num_layers=num_layers,
        batch_size=batch_size,
        input_size=(input_sequence_length,),
        loop_count=loop_count,
        data_format=data_format,
        total_time=total_time,
        total_samples=total_samples,
        evaluation_score=evaluation_score,
        optimization_level=optimization_level,
        trace_enabled=trace_enabled,
        model_info=full_model_name,
        display_name=display_name,
        input_is_image=False,
        input_sequence_length=input_sequence_length,
        experimental_weight_dtype=experimental_weight_dtype,
        **tt_xla_device_fields(),
    )

    return result
