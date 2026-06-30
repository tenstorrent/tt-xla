# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Z-Image — ZImageTransformer2DModel component test."""

import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.utilities.torch_multichip_utils import get_mesh

from third_party.tt_forge_models.z_image.pytorch import ModelLoader, ModelVariant
from third_party.tt_forge_models.z_image.pytorch.src.model_utils import (
    MESH_SHAPES,
    SEED,
)


@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.xfail(
    reason=(
        "RoPE complex legalization is fixed (tt-mlir #8874, merged), but the "
        "~6.2B DiT does not fit a single Wormhole (DRAM OOM). Compiles and "
        "passes on a single Blackhole; sharded multichip is the primary target."
    ),
    strict=False,
)
def test_transformer():
    _run(sharded=False)


@pytest.mark.model_test
@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.tensor_parallel
@pytest.mark.xfail(
    reason=(
        "RoPE complex legalization is fixed (tt-mlir #8874, merged); sharded "
        "(2,4) mesh now compiles + runs e2e. Remaining blockers: ttnn.concat "
        "L1 overflow at full depth (tt-xla #5367, needs tt-mlir #8860) and "
        "adaLN modulation sharding PCC drop (tt-xla #5351, needs "
        "tt-forge-models #783); model=4 PCC ~0.88 (<0.99)."
    ),
    strict=False,
)
def test_transformer_sharded():
    _run(sharded=True)


def _run(sharded: bool):
    xr.set_device_type("TT")
    torch.manual_seed(SEED)

    loader = ModelLoader(ModelVariant.TRANSFORMER)
    model = loader.load_model(dtype_override=torch.bfloat16)
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    mesh = None
    shard_spec_fn = None
    if sharded:
        num_devices = xr.global_runtime_device_count()
        if num_devices < 2:
            pytest.skip(
                f"test_transformer_sharded requires >= 2 TT devices, got {num_devices}"
            )
        if num_devices not in MESH_SHAPES:
            pytest.skip(
                f"Unsupported device count {num_devices}; "
                f"expected one of {sorted(MESH_SHAPES)}"
            )
        mesh_shape, mesh_names = loader.get_mesh_config(num_devices)
        mesh = get_mesh(mesh_shape, mesh_names)
        shard_spec_fn = loader.load_shard_spec

    run_graph_test(
        model,
        inputs,
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=shard_spec_fn,
    )
