# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""FLUX.1-dev — FluxTransformer2DModel component test (1024x1024 latent geometry).

The ~24 GB bf16 transformer fits on a single chip with the current tt-mlir/tt-metal
pin (no DRAM OOM), so the single-device test runs. The margin is thin, so the
sharded (4-way tensor-parallel) variant is kept as the durable, always-fitting
coverage. See loader.shard_transformer_specs for the sharding scheme.
"""

import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, RunMode, run_graph_test
from infra.utilities.torch_multichip_utils import get_mesh
from utils import BringupStatus, Category

from third_party.tt_forge_models.config import Parallelism
from third_party.tt_forge_models.flux.pytorch import ModelLoader, ModelVariant


@pytest.mark.single_device
@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=ModelLoader.get_model_info(ModelVariant.TRANSFORMER),
    parallelism=Parallelism.SINGLE_DEVICE,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_transformer():
    _run(sharded=False)


@pytest.mark.bh_galaxy
@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=ModelLoader.get_model_info(ModelVariant.TRANSFORMER),
    parallelism=Parallelism.TENSOR_PARALLEL,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
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

    run_graph_test(
        model,
        inputs,
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=shard_spec_fn,
    )
