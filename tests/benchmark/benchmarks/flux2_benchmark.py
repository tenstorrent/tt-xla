# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Benchmark core for the FLUX.2-dev pipeline components.

FLUX.2-dev is a rectified-flow text-to-image model made of three independently
loadable components (see third_party/tt_forge_models/flux2):

  - transformer  : Flux2Transformer2DModel (MM-DiT, ~32B)  -> tensor parallel
  - text_encoder : Mistral3 prompt embedder (~24B)         -> tensor parallel
  - vae          : AutoencoderKLFlux2 decoder (~0.1B)       -> single device

None of the existing benchmark helpers (test_llm / test_vision / test_encoder)
fit this model: the components take multiple positional tensors, the two large
ones must be sharded tensor-parallel across the mesh, and the VAE runs on a
single chip. This module provides a single component benchmark that handles all
three by reusing the validated multi-chip primitives from `tests/infra`
(SPMD enable, mesh build, weight sharding, tt compile) — it does not invent its
own timing or measurement logic beyond the standard benchmark reporting helpers
shared with the other benchmark files.
"""

import socket
import time

import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr

# Multi-chip primitives, shared with the runner tests (validated path).
from infra.utilities.torch_multichip_utils import enable_spmd, get_mesh
from utils import (
    align_arch,
    build_xla_export_name,
    compute_pcc,
    create_benchmark_result,
    get_benchmark_metadata,
    move_to_cpu,
    print_benchmark_results,
)

xr.set_device_type("TT")

MODULE_EXPORT_PATH = "modules"


def _get_device_arch():
    """SPMD-safe device arch lookup.

    Under SPMD, ``xm.xla_device()`` resolves to the virtual ``SPMD:0`` device
    which ``xla_device_kind`` cannot map, so the standard ``get_xla_device_arch``
    helper raises. Query the runtime device attributes directly instead.
    """
    try:
        attrs = xr.global_runtime_device_attributes()
        if attrs:
            return align_arch(str(attrs[0].get("device_arch", "")).lower())
    except Exception:
        pass
    return ""


def _move_inputs(inputs, device):
    """Move a list of positional input tensors to the device."""
    return [t.to(device) if isinstance(t, torch.Tensor) else t for t in inputs]


def _execute_and_measure(model, device_inputs, loop_count):
    """Run the component `loop_count` times and return (last_output_cpu, total_s).

    Each iteration is one component forward (for the DiT this is one denoise
    step). We block on the final result by moving it to CPU, mirroring the other
    benchmark files.
    """
    last_output = None
    start = time.perf_counter_ns()
    with torch.no_grad():
        outputs = []
        for i in range(loop_count):
            it_start = time.perf_counter_ns()
            out = model(*device_inputs)
            outputs.append(out)
            it_end = time.perf_counter_ns()
            print(f"Iteration {i} took {(it_end - it_start) / 1e6:.04} ms")
        # Block on device work by pulling results back to host.
        cpu_outputs = [move_to_cpu(o) for o in outputs]
        last_output = cpu_outputs[-1]
    end = time.perf_counter_ns()
    total_time = (end - start) / 1e9
    print(f"Total time: {total_time:.04}s for {loop_count} iterations")
    return last_output, total_time


def benchmark_flux2_component_torch_xla(
    loader,
    variant,
    model_info_name,
    display_name,
    optimization_level,
    trace_enabled,
    loop_count,
    warmup_steps,
    required_pcc,
    data_format=torch.bfloat16,
    custom_compile_options=None,
):
    """Benchmark a single FLUX.2-dev component on the TT backend.

    Tensor-parallel components (DiT, text encoder) are sharded across the mesh
    returned by the loader; the VAE runs single-chip. PCC is checked against a
    CPU golden run of the same component.

    Args:
        loader: A `flux2` ModelLoader instance bound to `variant`.
        variant: The ModelVariant being benchmarked.
        model_info_name / display_name: Names for reporting.
        optimization_level: tt-mlir optimization level (0/1/2).
        trace_enabled: Enable on-device trace.
        loop_count: Number of timed forward iterations.
        warmup_steps: Number of warmup iterations (first one triggers compile).
        required_pcc: Minimum PCC vs CPU golden.
        data_format: Model/input dtype.
        custom_compile_options: Extra tt compile options merged into the option
            dict (used by perf tuning, e.g. fp32_dest_acc_en / math_fidelity).
    """
    num_devices = xr.global_runtime_device_count()
    mesh_shape, mesh_names = loader.get_mesh_config(num_devices)
    sharded = mesh_shape is not None and any(d > 1 for d in mesh_shape)

    # On a multi-chip box every compile must carry an SPMD mesh that covers all
    # devices, otherwise the runtime wraps the graph into an implicit mesh and
    # aborts ("Bad StatusOr access: INTERNAL: Error code: 13"). Sharded
    # components (DiT, text encoder) use the loader's tensor-parallel mesh; the
    # single-chip VAE is run replicated across the full mesh (no mark_sharding).
    use_spmd = num_devices > 1
    if use_spmd and not sharded:
        mesh_shape, mesh_names = (1, num_devices), ("batch", "model")

    print(f"Running FLUX.2 component benchmark: {model_info_name}")
    print(f"""Configuration:
    variant={variant}
    optimization_level={optimization_level}
    trace_enabled={trace_enabled}
    loop_count={loop_count}
    warmup_steps={warmup_steps}
    data_format={data_format}
    required_pcc={required_pcc}
    num_devices={num_devices}
    mesh_shape={mesh_shape} (sharded={sharded})
    custom_compile_options={custom_compile_options}
    """)

    # --- Load component and inputs ---
    model = loader.load_model(dtype_override=data_format).eval()
    inputs = loader.load_inputs(dtype_override=data_format)

    # --- CPU golden (single forward) ---
    with torch.no_grad():
        golden_output = move_to_cpu(model(*inputs))

    # --- SPMD / mesh setup ---
    # Tensor-parallel components are sharded; the VAE is replicated. Both need a
    # full-device mesh + SPMD enabled on a multi-chip box.
    mesh = None
    if use_spmd:
        enable_spmd()
        mesh = get_mesh(mesh_shape, mesh_names)

    # --- Compile options ---
    export_model_name = build_xla_export_name(
        model_name=display_name,
        num_layers=None,
        batch_size=1,
        input_sequence_length=None,
    )
    options = {
        "optimization_level": optimization_level,
        "export_path": MODULE_EXPORT_PATH,
        "export_model_name": export_model_name,
        "enable_trace": trace_enabled,
    }
    if custom_compile_options:
        options.update(custom_compile_options)
    torch_xla.set_custom_compile_options(options)

    # Compile in place (lazy; actual lowering happens on first execution).
    model.compile(backend="tt")

    device = torch_xla.device()
    model = model.to(device)
    device_inputs = _move_inputs(inputs, device)

    # --- Shard weights across the mesh (tensor parallel only) ---
    if sharded:
        shard_specs = loader.load_shard_spec(model)
        if shard_specs:
            for tensor, shard_spec in shard_specs.items():
                xs.mark_sharding(tensor, mesh, shard_spec)
            print(f"Marked sharding on {len(shard_specs)} weight tensors.")

    # --- Warmup (first iteration triggers compile) ---
    print("Starting warmup...")
    warmup_count = min(warmup_steps, loop_count)
    if warmup_count > 0:
        _execute_and_measure(model, device_inputs, warmup_count)
    print("Warmup completed.")

    # --- Timed benchmark ---
    print("Starting benchmark...")
    last_output, total_time = _execute_and_measure(model, device_inputs, loop_count)
    print("Benchmark completed.")

    total_samples = loop_count
    samples_per_sec = total_samples / total_time

    metadata = get_benchmark_metadata()
    data_format_str = "bfloat16" if data_format == torch.bfloat16 else str(data_format)

    print_benchmark_results(
        model_title=model_info_name,
        full_model_name=model_info_name,
        model_type="Text-to-Image (FLUX.2-dev component), Random Input Data",
        dataset_name="Random Data",
        date=metadata["date"],
        machine_name=metadata["machine_name"],
        total_time=total_time,
        total_samples=total_samples,
        samples_per_sec=samples_per_sec,
        batch_size=1,
        data_format=data_format_str,
    )

    # --- PCC validation against CPU golden ---
    pcc_value = compute_pcc(golden_output, last_output)
    assert (
        pcc_value >= required_pcc
    ), f"PCC comparison failed. PCC={pcc_value:.6f}, Required={required_pcc}"
    print(f"PCC verification passed with PCC={pcc_value:.6f}")

    result = create_benchmark_result(
        full_model_name=model_info_name,
        model_type="Text-to-Image (FLUX.2-dev component), Random Input Data",
        dataset_name="Random Data",
        num_layers=-1,
        batch_size=1,
        input_size=(1, 1, 1),
        loop_count=loop_count,
        data_format=data_format_str,
        total_time=total_time,
        total_samples=total_samples,
        optimization_level=optimization_level,
        program_cache_enabled=True,
        trace_enabled=trace_enabled,
        model_info=model_info_name,
        display_name=display_name,
        torch_xla_enabled=True,
        backend="tt",
        device_name=socket.gethostname(),
        arch=_get_device_arch(),
        device_count=num_devices,
        mesh_shape=tuple(mesh_shape) if mesh_shape else None,
        input_is_image=False,
    )
    result["measurements"].append(
        {
            "iteration": 1,
            "step_name": model_info_name,
            "step_warm_up_num_iterations": warmup_steps,
            "measurement_name": "pcc",
            "value": pcc_value,
            "target": required_pcc,
            "device_power": -1.0,
            "device_temperature": -1.0,
        }
    )
    return result
