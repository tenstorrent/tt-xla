# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""HunyuanVideo — HunyuanVideoTransformer3DModel (DiT) component test."""

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.utilities.torch_multichip_utils import get_mesh

from third_party.tt_forge_models.hunyuan_video.pytorch import ModelLoader, ModelVariant


@pytest.mark.skip
def test_transformer():
    _run(sharded=False)


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

    # Keep matmul/linear operands in bf16 during device compilation instead of the
    # default f32 upcast. Otherwise the fused AdaLayerNorm modulation weight is
    # materialized in f32 (~3.4 GB) and OOMs DRAM. This only affects the TT
    # compile path; the CPU golden is unchanged.
    torch_options = {"tt_preserve_matmul_input_dtype": True}

    run_graph_test(
        model,
        inputs,
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=shard_spec_fn,
        torch_options=torch_options,
    )
