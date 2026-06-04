# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""HiDream-I1-Fast — Llama-3.1-8B-Instruct (text_encoder_4) component test. Params: ~8.0 B.

Llama-3.1-8B exceeds single-chip memory; only the sharded variant is expected
to run. The unsharded test stays in the file for reference and is skipped by
default.
"""

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.utilities.torch_multichip_utils import get_mesh

from third_party.tt_forge_models.hidream_i1.pytorch import ModelLoader, ModelVariant


@pytest.mark.skip(
    reason="OOM on single device — Llama-3.1-8B exceeds single-chip memory; sharded variant runs"
)
def test_text_encoder_4():
    _run(sharded=False)


@pytest.mark.nightly
@pytest.mark.model_test
def test_text_encoder_4_sharded():
    _run(sharded=True)


def _run(sharded: bool):
    xr.set_device_type("TT")
    torch.manual_seed(42)

    loader = ModelLoader(ModelVariant.TEXT_ENCODER_4)
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
