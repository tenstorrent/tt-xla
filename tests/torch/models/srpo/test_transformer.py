# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""SRPO (tencent/SRPO) — FLUX.1-dev fine-tune DiT transformer component test.

SRPO is a ~12B-parameter FLUX.1-dev fine-tune (a drop-in
``diffusers.FluxTransformer2DModel`` state dict) that runs out of DRAM on a
single chip. The sharded variant brings it up across a multi-chip mesh with
Megatron-1D tensor parallelism (tt-forge-models#748) and runs end-to-end on the
8-chip galaxy runner.
"""

import pytest

# Skip until the submodule uplift brings in the SRPO loader (tt-forge-models#698)
# and its sharding API (get_mesh_config / load_shard_spec, tt-forge-models#748).
pytest.importorskip(
    "third_party.tt_forge_models.srpo.pytorch.src.shard_specs",
    reason="SRPO loader + tensor-parallel sharding (tt-forge-models#698/#748) not yet in the submodule pin",
)

import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.utilities.torch_multichip_utils import get_mesh

from third_party.tt_forge_models.srpo.pytorch import ModelLoader, ModelVariant


@pytest.mark.skip(
    reason="model size ~12B — won't fit on a single chip; sharded variant runs"
)
def test_transformer():
    _run(sharded=False)


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.tensor_parallel
@pytest.mark.galaxy
def test_transformer_sharded():
    _run(sharded=True)


def _run(sharded: bool):
    xr.set_device_type("TT")
    torch.manual_seed(42)

    loader = ModelLoader(ModelVariant.BASE)
    model = loader.load_model(dtype_override=torch.float32)
    inputs = loader.load_inputs(dtype_override=torch.float32)

    mesh = None
    shard_spec_fn = None
    if sharded:
        mesh_shape, mesh_names = loader.get_mesh_config(
            xr.global_runtime_device_count()
        )
        mesh = get_mesh(mesh_shape, mesh_names)
        shard_spec_fn = loader.load_shard_spec

    run_graph_test(
        model,
        inputs,
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=shard_spec_fn,
    )
