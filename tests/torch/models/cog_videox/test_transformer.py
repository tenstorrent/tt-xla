# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CogVideoX-5b — CogVideoXTransformer3DModel (DiT) component test."""

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.testers.single_chip.model.torch_model_tester import _mask_jax_accelerator
from infra.utilities.torch_multichip_utils import get_mesh

from tests.infra.testers.compiler_config import CompilerConfig
from third_party.tt_forge_models.cog_videox.pytorch import ModelLoader, ModelVariant

# @pytest.mark.skip()
# def test_transformer():
#     _run(sharded=False)


# @pytest.mark.xfail(
#     reason="ttnn.concat: required CB page size (1732608 B) exceeds per-core L1 capacity (1395360 B); op cannot fit on this device."
# )
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
    if sharded:
        mesh_shape, mesh_names = loader.get_mesh_config(
            xr.global_runtime_device_count()
        )
        mesh = get_mesh(mesh_shape, mesh_names)
        shard_spec_fn = loader.load_shard_spec

        # TP (shard_spec_fn) shards weights on "model". SP on "batch" shards the
        # activation token (L) dim via forward-hook sharding constraints; register
        # before run_graph_test traces the block stack. Without this the per-block
        # CCLs run on the full L=1576 and overflow L1 (ttnn.concat page > capacity).
        loader.apply_activation_sharding(model, mesh)

    # At optimization_level=0 (default) no memory-layout optimization runs, so the
    # compiler const-evals the fused adaLN modulation weights (the 84 per-block
    # norm.linear matrices that all consume `temb`) into one giant row-major concat
    # -> tensor<512x1554432xf32>, whose per-page size exceeds per-core L1
    # (ttnn.concat CB page 3465216 B > 1395360 B). Raising the optimization level
    # lets the memory-layout pass relayout/place that concat so it fits. Mirrors
    # PR #4946 (VAE decoders set optimization_level to dodge a compile-time OOM).
    compiler_config = CompilerConfig(optimization_level=2)

    with _mask_jax_accelerator():
        run_graph_test(
            model,
            inputs,
            framework=Framework.TORCH,
            mesh=mesh,
            shard_spec_fn=shard_spec_fn,
            compiler_config=compiler_config,
        )
