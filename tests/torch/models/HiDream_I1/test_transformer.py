# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""HiDream-I1-Fast — HiDreamImageTransformer2DModel (Sparse-MoE MM-DiT) component test. Params: ~17 B static, ~14 B activated per token (top-2-of-4 MoE).

The transformer's 17 B parameters exceed single-chip memory; only the sharded
variant is expected to run. The unsharded test stays in the file for reference
and is skipped by default.
"""

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.utilities.torch_multichip_utils import get_mesh

from third_party.tt_forge_models.hidream_i1.pytorch import ModelLoader, ModelVariant


@pytest.mark.skip(
    reason="OOM on single device — HiDream DiT (17 B) exceeds single-chip memory; sharded variant runs"
)
def test_transformer():
    _run(sharded=False)


@pytest.mark.xfail(
    reason="Out of Memory: Not enough space to allocate 5872025600 B DRAM buffer across 12 banks, where each bank needs to store 489336832 B, but bank size is 1071821792 B - https://github.com/tenstorrent/tt-xla/issues/4761"
)
@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.llmbox
def test_transformer_sharded():
    _run(sharded=True)


def _run(sharded: bool):
    xr.set_device_type("TT")
    torch.manual_seed(42)

    loader = ModelLoader(ModelVariant.TRANSFORMER)
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
