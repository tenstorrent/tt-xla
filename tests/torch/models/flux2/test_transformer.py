# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""FLUX.2-dev — Flux2Transformer2DModel component test (128x128 latent geometry)."""

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.utilities.torch_multichip_utils import get_mesh

from tests.infra.testers.compiler_config import CompilerConfig
from third_party.tt_forge_models.flux2.pytorch import ModelLoader, ModelVariant


# @pytest.mark.skip(
#     reason="~32B transformer — exceeds single-chip DRAM; use test_transformer_sharded on 8+ chips"
# )
@pytest.mark.single_device
@pytest.mark.model_test
def test_transformer():
    _run(sharded=False)


# @pytest.mark.xfail(
#     reason="Out of Memory: cannot allocate 56623104 B DRAM buffer across 12 banks "
#     "(DRAM ~full, ~38 MB free) when sharded across 8 chips — "
#     "TT_FATAL bank_manager.cpp:462. 32B transformer is still DRAM-bound. "
#     "Tracking issue TBD."
# )
@pytest.mark.tensor_parallel
@pytest.mark.nightly
@pytest.mark.model_test
def test_transformer_sharded():
    _run(sharded=True)


def _run(sharded: bool):
    xr.set_device_type("TT")
    torch.manual_seed(42)

    loader = ModelLoader(ModelVariant.TRANSFORMER)
    model = loader.load_model(dtype_override=torch.bfloat16)
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    mesh = None
    shard_spec_fn = None
    compiler_config = None
    if sharded:
        mesh_shape, mesh_names = loader.get_mesh_config(
            xr.global_runtime_device_count()
        )
        mesh = get_mesh(mesh_shape, mesh_names)
        shard_spec_fn = loader.load_shard_spec
        # The ~32B transformer is 8.06 GB/device of bf16 weights when sharded
        # 8-way — too thin a margin on a 12.85 GB n300, so a mid-graph tilize
        # OOMs. Convert matmul/linear weights to block-float8 on device
        # (~4.3 GB/device), leaving headroom for activations + CCL buffers.
        compiler_config = CompilerConfig(experimental_weight_dtype="bfp_bf8")

    run_graph_test(
        model,
        inputs,
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=shard_spec_fn,
        compiler_config=compiler_config,
    )
