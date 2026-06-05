# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
HunyuanImage 2.1 (Distilled) — HunyuanImageTransformer2DModel (MM-DiT) test.

Hybrid MM-DiT: 20 dual-stream + 40 single-stream blocks, hidden 3584 (17.45B).
One denoising step: predicts the latent given timestep + dual text conditioning.

IN:  hidden_states (1, 64, 64, 64), timestep (1,), timestep_r (1,), guidance (1,),
     encoder_hidden_states (1, 1000, 3584), encoder_attention_mask (1, 1000) int64,
     encoder_hidden_states_2 (1, 128, 1472), encoder_attention_mask_2 (1, 128) int64
OUT: sample (1, 64, 64, 64)

weight_fit: tensor_parallel only — bf16 weights 34.9 GiB exceed every single-chip
budget (n150 10.2 GiB, p150 27.2 GiB). Runs sharded on a multichip mesh
(promotion via model-bringup-multichip). PCC 0.99.
"""

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig
from infra.utilities.torch_multichip_utils import get_mesh

from tests.infra.testers.compiler_config import CompilerConfig
from third_party.tt_forge_models.hunyuan_image_2_1.pytorch import (
    ModelLoader,
    ModelVariant,
)


@pytest.mark.skip(
    reason="17.45B — bf16 weights exceed every single-chip DRAM budget; "
    "tensor_parallel node test_transformer_sharded runs instead (weight_fit.json)"
)
def test_transformer():
    _run(sharded=False)


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.tensor_parallel
def test_transformer_sharded():
    _run(sharded=True)


def _run(sharded: bool):
    xr.set_device_type("TT")
    torch.manual_seed(42)
    compiler_config = CompilerConfig(optimization_level=1)

    # bf16 on the TP mesh: fp32 weights (69.8 GiB) OOM even 4-way (#4780);
    # bf16 (34.9 GiB) is ~8.7 GiB/chip on a 4-way mesh, within the n300 budget.
    # TP degree must divide num_attention_heads (28) — use 2 or 4, not 8.
    loader = ModelLoader(ModelVariant.TRANSFORMER)
    model = loader.load_model(dtype_override=torch.bfloat16)
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    mesh = None
    shard_spec_fn = None
    if sharded:
        xr.use_spmd()
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
        compiler_config=compiler_config,
        comparison_config=ComparisonConfig(pcc=PccConfig(required_pcc=0.99)),
    )
