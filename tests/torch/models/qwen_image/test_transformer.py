# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen-Image — QwenImageTransformer2DModel (MMDiT) component test (1024x1024)."""

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig
from infra.utilities.torch_multichip_utils import get_mesh

from third_party.tt_forge_models.qwen_image.pytorch import ModelLoader, ModelVariant


@pytest.mark.skip(
    reason="~20B transformer — weights (38 GiB bf16) exceed single-chip DRAM "
    "(~32 GiB, 119%); requires a multi-chip mesh. p300 (2-chip) is non-standard "
    "and the whole-model e2e target is qb2 (4-chip), so use test_transformer_sharded."
)
@pytest.mark.single_device
@pytest.mark.model_test
def test_transformer():
    _run(sharded=False)


@pytest.mark.tensor_parallel
@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.lb_blackhole
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
    comparison_config = ComparisonConfig()
    if sharded:
        mesh_shape, mesh_names = loader.get_mesh_config(
            xr.global_runtime_device_count()
        )
        mesh = get_mesh(mesh_shape, mesh_names)
        shard_spec_fn = loader.load_shard_spec
        # Megatron column→row tensor parallelism introduces a small bf16 gap vs
        # the CPU reference; relax the default 0.99 gate to 0.98 (tune in bring-up).
        comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.98))

    run_graph_test(
        model,
        inputs,
        framework=Framework.TORCH,
        comparison_config=comparison_config,
        mesh=mesh,
        shard_spec_fn=shard_spec_fn,
    )
