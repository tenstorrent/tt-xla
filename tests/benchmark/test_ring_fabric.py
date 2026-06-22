# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Minimal multichip test that exercises *only* the device + fabric open path the
LLM perf benchmark goes through.

Purpose: reproduce/guard the model-independent fabric-setup failure on
galaxy-wh-6u (tenstorrent/tt-xla#5210 — "Failed to add pinning constraints",
a topology-mapper TT_FATAL that happens during fabric setup, before any model
compile). Because the failure is model-independent, we don't need a real model,
perf measurement, or PCC check — we only need to drive the device + fabric open
path and let the sharded program execute.

How the perf benchmark opens device + fabric (see
``tests/benchmark/benchmarks/llm_benchmark.py``):

    xr.set_device_type("TT")
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"   # multichip only
    xr.use_spmd()                                # multichip only
    device = torch_xla.device()                  # connects the device
    mesh = Mesh(range(N), mesh_shape, axis_names) # device mesh
    # ... mark_sharding + torch.compile(backend="tt") + execute

The fabric is selected inside the PJRT plugin when the mesh device opens:
``ClientInstance::openMeshDevice`` -> ``computeFabricConfig`` ->
``tt::runtime::computeMeshFabricConfig``. Each mesh axis is classified
independently; on a Wormhole galaxy both axes wrap, so a 2D mesh opens a torus
(per-axis topology ``[Ring, Ring]``). Executing a sharded all-reduce forces the
mesh device + fabric to actually open, which is the path that currently
TT_FATALs on galaxy-wh-6u.

Mesh shapes:
  - galaxy-wh-6u (32 devices) -> 4x8 (full torus)
  - n300-llmbox               -> 2x2 sub-mesh
"""

import numpy as np
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra.utilities.torch_multichip_utils import enable_spmd
from torch_xla.distributed.spmd import Mesh

xr.set_device_type("TT")

GALAXY_MESH_SHAPE = (4, 8)  # galaxy-wh-6u: full 32-device torus
LLMBOX_MESH_SHAPE = (2, 2)  # n300-llmbox: 2x2 sub-mesh


def test_ring_fabric_open():
    """
    Open the device + 2D (torus) fabric exactly like the perf benchmark and run
    a single sharded all-reduce so the fabric is actually brought up. The test
    passes if the device/fabric open path completes without erroring.
    """
    # Mirror the benchmark's multichip device-open sequence.
    enable_spmd()  # sets CONVERT_SHLO_TO_SHARDY=1 and xr.use_spmd()
    device = torch_xla.device()

    num_devices = xr.global_runtime_device_count()
    mesh_shape = GALAXY_MESH_SHAPE if num_devices >= 32 else LLMBOX_MESH_SHAPE
    mesh_devices = mesh_shape[0] * mesh_shape[1]
    assert num_devices >= mesh_devices, (
        f"Need at least {mesh_devices} devices for a {mesh_shape} mesh, "
        f"found {num_devices}."
    )

    mesh = Mesh(np.arange(mesh_devices), mesh_shape, ("x", "y"))
    print(f"Created device mesh: {mesh_shape} using {mesh_devices}/{num_devices} devices.")

    # Tiny matmul whose contraction dim is sharded across BOTH mesh axes, so the
    # result requires an all-reduce that traverses both ring axes of the torus.
    # ``hidden`` (256) is divisible by both mesh sizes (32 and 4).
    batch, hidden, out = 32, 256, 128

    class ShardedMM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.randn(hidden, out, dtype=torch.bfloat16))

        def forward(self, x):
            return torch.matmul(x, self.w)

    model = ShardedMM().eval().to(device)
    activation = torch.randn(batch, hidden, dtype=torch.bfloat16).to(device)

    xs.mark_sharding(activation, mesh, (None, ("x", "y")))
    xs.mark_sharding(model.w, mesh, (("x", "y"), None))

    torch_xla.set_custom_compile_options({"optimization_level": 0})
    compiled = torch.compile(model, backend="tt")

    # Executing the sharded all-reduce forces the mesh device + fabric to open.
    result = compiled(activation)
    torch_xla.sync()
    # Block until execution completes; we only care that it ran, not the values.
    result.to("cpu")


def test_ring_fabric_open_light():
    """
    Lighter-weight variant of :func:`test_ring_fabric_open`.

    The #5210 topology-mapper TT_FATAL happens during device + fabric open (the
    mesh device is created when the PJRT client initializes the system
    descriptor), i.e. *before* any program executes. This variant therefore
    exercises the exact same device + 2D (torus) fabric open path, but runs a
    *sharded elementwise* op instead of a sharded all-reduce. An elementwise op
    needs no cross-device communication, so it compiles no CCL (all-gather /
    all-reduce) kernels -- which on a cold kernel cache can take many minutes --
    while still opening the mesh device + fabric and executing a sharded program
    across the full torus.

    Use this as the fast fabric-open regression guard; ``test_ring_fabric_open``
    additionally validates that fabric CCLs lower and run.
    """
    enable_spmd()  # sets CONVERT_SHLO_TO_SHARDY=1 and xr.use_spmd()
    device = torch_xla.device()

    num_devices = xr.global_runtime_device_count()
    mesh_shape = GALAXY_MESH_SHAPE if num_devices >= 32 else LLMBOX_MESH_SHAPE
    mesh_devices = mesh_shape[0] * mesh_shape[1]
    assert num_devices >= mesh_devices, (
        f"Need at least {mesh_devices} devices for a {mesh_shape} mesh, "
        f"found {num_devices}."
    )

    mesh = Mesh(np.arange(mesh_devices), mesh_shape, ("x", "y"))
    print(f"Created device mesh: {mesh_shape} using {mesh_devices}/{num_devices} devices.")

    # ``batch`` (32) is divisible by mesh axis x (4 and 2); ``hidden`` (256) is
    # divisible by mesh axis y (8 and 2). The activation is sharded across BOTH
    # mesh axes so the program runs on every device of the torus.
    batch, hidden = 32, 256

    class ShardedElementwise(torch.nn.Module):
        def forward(self, x):
            return torch.relu(x) * 2.0

    model = ShardedElementwise().eval().to(device)
    activation = torch.randn(batch, hidden, dtype=torch.bfloat16).to(device)

    xs.mark_sharding(activation, mesh, ("x", "y"))

    torch_xla.set_custom_compile_options({"optimization_level": 0})
    compiled = torch.compile(model, backend="tt")

    # Executing the sharded program forces the mesh device + fabric to open.
    # No reduction across shards => no CCL kernels to compile.
    compiled(activation)
    torch_xla.sync()
