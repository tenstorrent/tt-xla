# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Minimal perf benchmark that exercises *only* the device + fabric open path the
LLM perf benchmark goes through, and forces a 2D torus fabric to be opened.

How the perf benchmark opens device + fabric (see
``tests/benchmark/benchmarks/llm_benchmark.py``):

    xr.set_device_type("TT")
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"   # multichip only
    xr.use_spmd()                                # multichip only
    device = torch_xla.device()                  # connects the device
    mesh = Mesh(range(N), mesh_shape, axis_names) # device mesh
    # ... mark_sharding + torch.compile(backend="tt") + execute

The fabric is selected inside the PJRT plugin when the mesh device is opened:
``ClientInstance::openMeshDevice`` -> ``computeFabricConfig`` ->
``tt::runtime::computeMeshFabricConfig`` (see
``pjrt_implementation/src/api/client_instance.cc`` and
``runtime/lib/common/mesh_fabric_config.cpp``). Each mesh axis is classified
independently: an axis whose lines wrap around becomes ``FABRIC_1D_RING``,
otherwise ``FABRIC_1D``. The per-axis configs map to per-axis mesh topology
(``Ring``/``Linear``) in ``module_builder.cc``, so:

  - a 1D ``(1, N)`` mesh -> ``[Disabled, Ring]``         (a single 1D ring)
  - a 2D ``(R, C)`` mesh  -> ``[Ring, Ring]``            (a torus)

To actually open and exercise a torus we therefore need a 2D mesh with both
axes > 1 (and hardware that wraps on both axes, i.e. a Wormhole galaxy). This
test builds such a mesh and shards the matmul's contraction dim across *both*
axes, so the resulting all-reduce traverses both ring dimensions of the torus.

Note: on the 32-device Blackhole galaxy (UBB), ``computeFabricConfig`` currently
force-overrides to ``FABRIC_1D`` (RING_RING is rejected as TORUS_XY), so the
torus auto-detection is meaningful on the Wormhole galaxy.
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


def _factor_2d(n: int) -> tuple:
    """Largest near-square 2D factorization (rows, cols) with rows <= cols."""
    best = 1
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            best = i
    return (best, n // best)


def test_ring_fabric_all_reduce(output_file):
    """
    Open device + torus fabric the same way the perf benchmark does, then run a
    single all-reduce that traverses both ring axes of the torus, verify it
    matches CPU, and record a minimal perf result (all-reduce iterations/sec).
    """
    # Mirror the benchmark's multichip device-open sequence.
    enable_spmd()  # sets CONVERT_SHLO_TO_SHARDY=1 and xr.use_spmd()
    device = torch_xla.device()

    num_devices = xr.global_runtime_device_count()
    assert num_devices > 1, (
        f"Fabric needs >1 device, found {num_devices}. "
        "This benchmark targets multi-chip systems (llmbox / wh galaxy / bh galaxy)."
    )

    # 2D mesh over all devices. Both axes > 1 so the runtime can classify each
    # axis as a ring and open a torus ([Ring, Ring]) on hardware that wraps on
    # both axes (Wormhole galaxy).
    rows, cols = _factor_2d(num_devices)
    assert rows > 1 and cols > 1, (
        f"Torus needs a 2D mesh with both axes > 1; device count {num_devices} "
        f"factors to {(rows, cols)}. Run on a system whose device count is composite "
        "(e.g. 8 -> 2x4, 32 -> 4x8)."
    )
    mesh_shape = (rows, cols)
    mesh = Mesh(np.arange(num_devices), mesh_shape, ("x", "y"))
    print(f"Created device mesh: {mesh_shape} with {num_devices} devices.")

    # Matmul whose contraction dim is sharded across BOTH mesh axes. The partial
    # products must be summed via an all-reduce that traverses both ring axes of
    # the torus. ``hidden`` must be divisible by rows*cols (== num_devices).
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

    # Shard the contraction dim of both operands across both axes ("x", "y").
    # Reducing over a dim sharded on both axes => all-reduce over the full torus.
    xs.mark_sharding(activation, mesh, (None, ("x", "y")))
    xs.mark_sharding(model.w, mesh, (("x", "y"), None))

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
