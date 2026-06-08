# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""HiDream-I1-Fast — T5-XXL encoder (text_encoder_3) component test. Params: ~4.6 B."""

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.utilities.torch_multichip_utils import get_mesh

from third_party.tt_forge_models.hidream_i1.pytorch import ModelLoader, ModelVariant


@pytest.mark.skip(
    reason="OOM on single device - https://github.com/tenstorrent/tt-xla/issues/4760 , sharded variant runs"
)
def test_text_encoder_3():
    _run(sharded=False)


@pytest.mark.xfail(
    reason="AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.9873067700897922. Required: pcc=0.99. - https://github.com/tenstorrent/tt-xla/issues/4847"
)
@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.llmbox
def test_text_encoder_3_sharded():
    _run(sharded=True)


def _run(sharded: bool):
    xr.set_device_type("TT")
    torch.manual_seed(42)

    loader = ModelLoader(ModelVariant.TEXT_ENCODER_3)
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
