# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Minimal perf benchmark that exercises *only* the device + fabric open path the
LLM perf benchmark goes through, and forces a ring fabric to be opened.

How the perf benchmark opens device + fabric (see
``tests/benchmark/benchmarks/llm_benchmark.py``):

    xr.set_device_type("TT")
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"   # multichip only
    xr.use_spmd()                                # multichip only
    device = torch_xla.device()                  # connects the device
    mesh = Mesh(range(N), mesh_shape, axis_names) # device mesh
    # ... mark_sharding + torch.compile(backend="tt") + execute

The fabric itself is selected inside the PJRT plugin when the mesh device is
opened: ``ClientInstance::openMeshDevice`` -> ``computeFabricConfig`` ->
``tt::runtime::computeMeshFabricConfig`` (see
``pjrt_implementation/src/api/client_instance.cc``). On a Wormhole galaxy the
mesh axis has a wrap-around ethernet connection, so the runtime auto-detects
``FABRIC_1D_RING`` and calls ``setFabricConfig(FABRIC_1D_RING)``.

This test does the minimum needed to drive that path: open the device, build a
1D mesh over the model axis, and run a single row-parallel matmul whose
contraction dim is sharded. That lowers to an all-reduce across the ring axis,
which requires the ring fabric to be up.
"""

import json
import os
import time

import numpy as np
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra.utilities.torch_multichip_utils import enable_spmd
from torch_xla.distributed.spmd import Mesh
from utils import (
    create_benchmark_result,
    get_benchmark_metadata,
    get_xla_device_arch,
    print_benchmark_results,
)

xr.set_device_type("TT")

# Number of timed all-reduce iterations (after a warmup iteration).
ITERATIONS = 16

DISPLAY_NAME = "ring_fabric_all_reduce"


def test_ring_fabric_all_reduce(output_file):
    """
    Open device + ring fabric the same way the perf benchmark does, then run a
    single all-reduce over the ring axis, verify it matches CPU, and record a
    minimal perf result (all-reduce iterations/sec).
    """
    # Mirror the benchmark's multichip device-open sequence.
    enable_spmd()  # sets CONVERT_SHLO_TO_SHARDY=1 and xr.use_spmd()
    device = torch_xla.device()

    num_devices = xr.global_runtime_device_count()
    assert num_devices > 1, (
        f"Ring fabric needs >1 device, found {num_devices}. "
        "This benchmark targets multi-chip systems (llmbox / wh galaxy / bh galaxy)."
    )

    # 1D mesh over all devices; the "model" axis forms the ring.
    mesh_shape = (1, num_devices)
    mesh = Mesh(np.arange(num_devices), mesh_shape, ("batch", "model"))
    print(f"Created device mesh: {mesh_shape} with {num_devices} devices.")

    # Row-parallel matmul: contraction dim sharded along "model" -> all-reduce.
    # ``hidden`` must be divisible by the model-axis size for an even shard.
    batch, hidden, out = 32, 64 * num_devices, 128

    class RowParallelMM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.randn(hidden, out, dtype=torch.bfloat16))

        def forward(self, x):
            return torch.matmul(x, self.w)

    model = RowParallelMM().eval()
    activation = torch.randn(batch, hidden, dtype=torch.bfloat16)
    golden = model(activation)

    model = model.to(device)
    activation = activation.to(device)

    # Shard the contraction dim of both operands across the ring axis. The
    # partial products must be summed via an all-reduce that traverses the ring.
    xs.mark_sharding(activation, mesh, ("batch", "model"))
    xs.mark_sharding(model.w, mesh, ("model", None))

    torch_xla.set_custom_compile_options({"optimization_level": 0})
    compiled = torch.compile(model, backend="tt")

    # Warmup (triggers compile + opens the ring fabric).
    out_warmup = compiled(activation)
    torch_xla.sync()
    _ = out_warmup.to("cpu")

    # Timed all-reduce iterations.
    start = time.perf_counter()
    last = None
    for _ in range(ITERATIONS):
        last = compiled(activation)
    torch_xla.sync()
    result_cpu = last.to("cpu")
    total_time = time.perf_counter() - start

    iters_per_sec = ITERATIONS / total_time if total_time > 0 else 0.0

    # Correctness: all-reduce result must match the unsharded CPU matmul.
    torch.testing.assert_close(
        result_cpu.to(torch.float32),
        golden.to(torch.float32),
        rtol=0.05,
        atol=0.05,
    )

    metadata = get_benchmark_metadata()
    arch = get_xla_device_arch()

    print_benchmark_results(
        model_title=DISPLAY_NAME,
        full_model_name=DISPLAY_NAME,
        model_type="ccl",
        dataset_name="Random Data",
        date=metadata["date"],
        machine_name=metadata["machine_name"],
        total_time=total_time,
        total_samples=ITERATIONS,
        samples_per_sec=iters_per_sec,
        batch_size=batch,
        data_format="bfloat16",
        input_sequence_length=-1,
        ttft_ms=0.0,
    )

    if output_file:
        results = create_benchmark_result(
            full_model_name=DISPLAY_NAME,
            model_type="ccl",
            dataset_name="Random Data",
            num_layers=1,
            batch_size=batch,
            input_size=(batch, hidden),
            loop_count=ITERATIONS,
            data_format="bfloat16",
            total_time=total_time,
            total_samples=ITERATIONS,
            custom_measurements=[
                {"measurement_name": "all_reduce_iters_per_sec", "value": iters_per_sec},
            ],
            optimization_level=0,
            program_cache_enabled=True,
            trace_enabled=False,
            model_info=DISPLAY_NAME,
            display_name=DISPLAY_NAME,
            torch_xla_enabled=True,
            backend="tt",
            device_name=metadata["machine_name"],
            arch=arch,
            input_is_image=False,
            input_sequence_length=-1,
            device_count=num_devices,
            mesh_shape=mesh_shape,
        )
        results["project"] = "tt-forge/tt-xla"
        results["model_rawname"] = DISPLAY_NAME
        with open(output_file, "w") as file:
            json.dump(results, file, indent=2)
