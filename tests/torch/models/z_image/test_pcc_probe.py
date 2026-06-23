# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TEMP PCC probe: is the sharded transformer PCC drop a shard-spec
divisibility bug (n_heads=30) or an op-lowering bug?

n_heads=30. model=4 -> 30%4!=0 (mid-head split, expected wrong).
model=2 -> 30%2==0 (whole heads). If model=2 PCC recovers, it's the shard spec.
Reduced resolution so it fits/runs fast; same ops/sharding structure.
"""
import numpy as np
import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from torch_xla.distributed.spmd import Mesh

from tests.infra.testers.compiler_config import CompilerConfig
import third_party.tt_forge_models.z_image.pytorch.src.model_utils as mu
from third_party.tt_forge_models.z_image.pytorch import ModelLoader, ModelVariant

# shrink resolution (latent 32x32 -> short seq) so it fits small device counts
mu.LATENT_H, mu.LATENT_W = 32, 32
NLAYERS = 2  # truncate the 30-layer DiT so the unsharded model fits one chip


def _load_small():
    loader = ModelLoader(ModelVariant.TRANSFORMER)
    model = loader.load_model(dtype_override=torch.bfloat16)
    t = model.transformer
    t.layers = t.layers[:NLAYERS]  # keep RoPE/attention; drop most blocks
    return loader, model


def _run(mesh_names, compiler_config=None):  # physical 2x4; "model" axis size by name order
    xr.set_device_type("TT")
    torch.manual_seed(mu.SEED)
    n = xr.global_runtime_device_count()
    loader, model = _load_small()
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)
    mesh = Mesh(np.array(range(n)), (2, 4), mesh_names)
    run_graph_test(model, inputs, framework=Framework.TORCH, mesh=mesh,
                   shard_spec_fn=loader.load_shard_spec,
                   compiler_config=compiler_config)


@pytest.mark.nightly
@pytest.mark.single_device
def test_pcc_single():  # unsharded baseline (truncated model fits one chip)
    xr.set_device_type("TT")
    torch.manual_seed(mu.SEED)
    loader, model = _load_small()
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)
    run_graph_test(model, inputs, framework=Framework.TORCH)


@pytest.mark.nightly
@pytest.mark.llmbox
def test_pcc_model2():  # model axis = first (size 2); 30 % 2 == 0 -> whole heads
    _run(("model", "batch"))


@pytest.mark.nightly
@pytest.mark.llmbox
def test_pcc_model4():  # model axis = second (size 4); 30 % 4 != 0 -> mid-head split
    _run(("batch", "model"))


@pytest.mark.nightly
@pytest.mark.llmbox
def test_pcc_model4_fp32acc():  # same as model4 but fp32 accumulation in matmuls/CCLs
    _run(("batch", "model"), compiler_config=CompilerConfig(fp32_dest_acc_en=True))
