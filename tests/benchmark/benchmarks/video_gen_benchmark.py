# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Benchmark harness for video-generation model components (Wan 2.2 TI2V-5B and
A14B: UMT5 text encoder, 3D Causal VAE encoder/decoder, WanDiT transformer).

Unlike the vision/encoder harnesses these components:
  - take a *list* of input tensors (the DiT takes 3, UMT5 takes 2),
  - run under SPMD tensor/sequence-parallel sharding across a multi-device
    mesh (some, like UMT5, OOM on a single device),
  - may need a custom execution context (``safe_xla_slicing`` for the VAE
    decoder), a custom compile/run context (``torch_function_override_disabled``
    for the DiT), and per-component compiler options.

Sharding is fully delegated to an ``apply_sharding_fn(wrapper, mesh)`` callback
so the harness stays agnostic to whether a component shards only weights (VAE,
UMT5) or weights plus activation hooks (DiT sequence parallelism).
"""

# Built-in modules
import socket
import time
from contextlib import nullcontext

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from infra.utilities.torch_multichip_utils import enable_spmd
from utils import (
    build_xla_export_name,
    compute_pcc,
    create_benchmark_result,
    get_benchmark_metadata,
    get_xla_device_arch,
    print_benchmark_results,
)

xr.set_device_type("TT")

WARMUP_STEPS = 2
EXECUTION_STEPS = 3

# Where the compiler dumps stablehlo/ttir/ttnn + flatbuffer artifacts; the perf
# CI collects them from here (matches the vision/encoder/llm harnesses).
MODULE_EXPORT_PATH = "modules"


def _execute_and_measure(
    model, inputs, loop_count, extract_output_tensor_fn, run_context
):
    """Run ``model(*inputs)`` ``loop_count`` times.

    Returns ``(last_output_cpu, total_time_s, per_forward_ms)`` where
    ``per_forward_ms`` is the wall-clock of each individual forward pass. Each
    iteration syncs the result back to host (``.to("cpu")``) so the measured
    time reflects full device round-trip latency.
    """
    last_output = None
    per_forward_ms = []
    start_time = time.perf_counter_ns()
    with torch.no_grad():
        for i in range(loop_count):
            start_iteration_time = time.perf_counter_ns()
            with run_context():
                output = model(*inputs)
                output = extract_output_tensor_fn(output)
                last_output = output.to("cpu")
            iteration_ms = (time.perf_counter_ns() - start_iteration_time) / 1e6
            per_forward_ms.append(iteration_ms)
            print(f"Iteration {i} took {iteration_ms:.04} ms")
    total_time = (time.perf_counter_ns() - start_time) / 1e9
    print(f"Total time: {total_time:.04}s for {loop_count} iterations")
    return last_output, total_time, per_forward_ms


def benchmark_video_gen_torch_xla(
    *,
    wrapper,
    inputs,
    model_info_name,
    display_name,
    compiler_config,
    ttnn_perf_metrics_output_file,
    sharded,
    mesh_fn=None,
    apply_sharding_fn=None,
    run_context=nullcontext,
    compile_context=nullcontext,
    extract_output_tensor_fn=lambda o: o,
    loop_count=EXECUTION_STEPS,
    warmup_steps=WARMUP_STEPS,
    data_format="bfloat16",
    required_pcc=0.97,
):
    """Benchmark a single video-generation model component with torch-xla.

    Builds a CPU golden output, compiles the wrapper for the Tenstorrent
    backend (optionally under SPMD sharding), runs warmup + timed iterations,
    and validates correctness via PCC.

    Args:
        wrapper: ``nn.Module`` (eval, bf16) on CPU that returns a single tensor.
        inputs: list of CPU input tensors passed positionally to ``wrapper``.
        model_info_name: Full model name for reporting.
        display_name: Short name used for perf-metric file naming.
        compiler_config: ``CompilerConfig`` providing the tt-mlir compile options.
        ttnn_perf_metrics_output_file: Base name for TTNN perf-metric JSON files.
        sharded: Whether to run under SPMD sharding.
        mesh_fn: Zero-arg callable returning the SPMD ``Mesh`` (built when sharded).
        apply_sharding_fn: ``fn(wrapper_on_device, mesh) -> None`` that applies all
            sharding for the component — ``xs.mark_sharding`` on weights and, for the
            DiT, ``apply_dit_sp_activation_sharding`` hooks. Called only when the mesh
            has >1 device.
        run_context: Zero-arg callable returning a context manager wrapping each
            forward call (e.g. ``safe_xla_slicing``); defaults to ``nullcontext``.
        compile_context: Zero-arg callable returning a context manager wrapping the
            whole compile + execution region (e.g. ``torch_function_override_disabled``
            for the DiT); defaults to ``nullcontext``.
        extract_output_tensor_fn: Extracts the tensor from the wrapper output.
        loop_count: Number of timed iterations.
        warmup_steps: Number of (compile + run) warmup iterations.
        data_format: Precision string for reporting ("bfloat16" / "float32").
        required_pcc: Minimum PCC threshold (asserted). If None, the CPU golden
            and PCC check are skipped entirely (perf-only run).

    Returns:
        Standardized benchmark result dictionary.
    """
    xr.set_device_type("TT")

    # CPU golden for the PCC check (computed before moving the wrapper to device,
    # so no deepcopy of multi-billion-parameter modules is needed, and the
    # sharded device weights never have to be all-gathered back to host).
    # Skipped entirely when required_pcc is None: the reference is a full CPU
    # forward of the model, which is impractical for the largest configs (e.g.
    # the 14B DiT at 720p), so a perf-only run can opt out of it.
    golden_output = None
    if required_pcc is not None:
        with torch.no_grad():
            golden_output = (
                extract_output_tensor_fn(wrapper(*inputs)).detach().to("cpu")
            )

    # Build mesh and enable SPMD before any XLA op when sharding.
    use_sharding = False
    mesh = None
    if sharded:
        mesh = mesh_fn()
        use_sharding = len(mesh.device_ids) > 1
        if use_sharding:
            enable_spmd()

    # Compile options: per-component CompilerConfig, plus the module export and
    # TTNN perf-metric capture the perf CI relies on (the export_path/name set
    # here override any from the CompilerConfig so artifacts land in ./modules).
    batch_size = int(inputs[0].shape[0])
    options = compiler_config.to_torch_compile_options()
    options["export_path"] = MODULE_EXPORT_PATH
    options["export_model_name"] = build_xla_export_name(
        model_name=display_name,
        num_layers=None,
        batch_size=batch_size,
        input_sequence_length=None,
    )
    options["ttnn_perf_metrics_enabled"] = True
    options["ttnn_perf_metrics_output_file"] = ttnn_perf_metrics_output_file
    torch_xla.set_custom_compile_options(options)

    device = xm.xla_device()

    # Move wrapper + inputs to device first; sharding must see XLA tensors.
    wrapper_on_device = wrapper.to(device)
    if hasattr(wrapper_on_device, "tie_weights"):
        wrapper_on_device.tie_weights()
    inputs_on_device = [t.to(device) for t in inputs]

    if use_sharding:
        assert apply_sharding_fn is not None, "sharded run requires apply_sharding_fn"
        apply_sharding_fn(wrapper_on_device, mesh)

    # The DiT needs the torch-function override disabled across both compile and
    # every forward, so the whole region runs inside compile_context.
    with compile_context():
        compiled = torch.compile(wrapper_on_device, backend="tt")

        print("Starting warmup...")
        _execute_and_measure(
            model=compiled,
            inputs=inputs_on_device,
            loop_count=warmup_steps,
            extract_output_tensor_fn=extract_output_tensor_fn,
            run_context=run_context,
        )
        print("Warmup completed.")

        print("Starting benchmark...")
        last_output, total_time, per_forward_ms = _execute_and_measure(
            model=compiled,
            inputs=inputs_on_device,
            loop_count=loop_count,
            extract_output_tensor_fn=extract_output_tensor_fn,
            run_context=run_context,
        )
        print("Benchmark completed.")

    total_samples = batch_size * loop_count
    samples_per_sec = total_samples / total_time
    input_size = tuple(inputs[0].shape)

    # Per-forward-pass latencies — the meaningful metric for video generation,
    # where throughput-style samples/sec is not informative. One measurement per
    # timed forward pass (warmup excluded, since its first pass includes compile).
    forward_pass_measurements = [
        {"measurement_name": "forward_pass_time_ms", "value": ms, "iteration": i + 1}
        for i, ms in enumerate(per_forward_ms)
    ]

    # Validate correctness (asserts internally on PCC < required_pcc) and record
    # the measured PCC as the evaluation score, matching the encoder harness.
    # When the golden was skipped (required_pcc is None) there is nothing to
    # check and no score is recorded.
    evaluation_score = None
    if required_pcc is not None:
        evaluation_score = compute_pcc(
            last_output, golden_output, required_pcc=required_pcc
        )
        print(f"PCC verification passed with PCC={evaluation_score:.6f}")
    else:
        print("PCC check skipped (required_pcc=None).")

    metadata = get_benchmark_metadata()
    model_type = "Video Generation, Random Input Data"
    dataset_name = "Random Data"
    num_layers = -1

    device_count = xr.global_runtime_device_count()
    # mesh.mesh_shape is the size tuple e.g. (1, 4); mesh.shape() is an
    # OrderedDict keyed by axis name, so tuple(...) of it would give the names.
    mesh_shape = tuple(mesh.mesh_shape) if use_sharding else None

    print_benchmark_results(
        model_title=model_info_name,
        full_model_name=model_info_name,
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
        input_size=input_size,
    )

    result = create_benchmark_result(
        full_model_name=model_info_name,
        model_type=model_type,
        dataset_name=dataset_name,
        num_layers=num_layers,
        batch_size=batch_size,
        input_size=input_size,
        loop_count=loop_count,
        data_format=data_format,
        total_time=total_time,
        total_samples=total_samples,
        evaluation_score=evaluation_score,
        custom_measurements=forward_pass_measurements,
        optimization_level=compiler_config.optimization_level,
        program_cache_enabled=True,
        trace_enabled=compiler_config.enable_trace,
        model_info=model_info_name,
        display_name=display_name,
        torch_xla_enabled=True,
        backend="tt",
        device_name=socket.gethostname(),
        arch=get_xla_device_arch(),
        device_count=device_count,
        mesh_shape=mesh_shape,
        # Inputs are text (2-D) / video (5-D) tensors, not 2-D images, so skip
        # the channels x H x W "image_dimension" formatting.
        input_is_image=False,
    )

    return result
