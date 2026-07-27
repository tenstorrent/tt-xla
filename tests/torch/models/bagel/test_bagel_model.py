# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""BAGEL — bagel_model Qwen2 MoT backbone (text path) tensor-parallel component test.

The bagel_model is a Qwen2-7B-scale Mixture-of-Transformers (~27 GiB bf16), weight-bound on a
single chip. It runs 4-way tensor parallel (28 attention heads => 8-way invalid; 4-way is the
largest head-divisible degree) on the Wormhole fabric, with a clean fixed-shape forward over the
understanding ("und") text path. See third_party/tt_forge_models/bagel/pytorch/loader.py.
"""

import inspect

import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, RunMode, run_graph_test
from infra.utilities.torch_multichip_utils import get_mesh
from utils import BringupStatus, Category

import third_party.tt_forge_models.bagel.pytorch.loader as bagel_loader
from tests.runner.requirements import RequirementsManager
from third_party.tt_forge_models.bagel.pytorch import ModelLoader, ModelVariant
from third_party.tt_forge_models.config import Parallelism

MODEL_INFO = ModelLoader._get_model_info(ModelVariant.BAGEL_MODEL)


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.llmbox
@pytest.mark.tensor_parallel
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    parallelism=Parallelism.TENSOR_PARALLEL,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_bagel_model_sharded():
    loader_path = inspect.getsourcefile(bagel_loader)
    with RequirementsManager.for_loader(loader_path, framework="torch"):
        xr.set_device_type("TT")
        torch.manual_seed(42)

        loader = ModelLoader(ModelVariant.BAGEL_MODEL)
        model = loader.load_model(dtype_override=torch.bfloat16)
        inputs = loader.load_inputs(seq_len=loader.DEFAULT_SEQ_LEN)

        mesh_shape, mesh_names = loader.get_mesh_config(
            xr.global_runtime_device_count()
        )
        mesh = get_mesh(mesh_shape, mesh_names)

        run_graph_test(
            model,
            [inputs],
            framework=Framework.TORCH,
            mesh=mesh,
            shard_spec_fn=loader.load_shard_spec,
        )
