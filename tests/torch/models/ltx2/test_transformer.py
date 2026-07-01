# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""LTX-2 — LTX2VideoTransformer3DModel (dual-stream DiT, ~19B) tensor-parallel component test."""

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.utilities.torch_multichip_utils import get_mesh

from third_party.tt_forge_models.ltx2.pytorch import ModelLoader, ModelVariant


@pytest.mark.xfail(
    reason="tt/torch_xla cannot lower the LTX-2 rope head-merge: "
    "apply_split_rotary_emb (transformer_ltx2.py) does "
    "out.swapaxes(1, 2).reshape(b, t, -1) and _assert_tensor_metadata rejects "
    "reshaping the transposed (non-contiguous) tensor: 'Cannot view a tensor "
    "with shape [1,128,30,128] strides (491520,128,16384,1) as (1,128,3840)'. "
    ".contiguous() is a no-op on XLA lazy tensors, so it is not fixable in the "
    "loader. Same root cause as the connectors component — "
    "https://github.com/tenstorrent/tt-xla/issues/5196"
)
@pytest.mark.model_test
@pytest.mark.tensor_parallel
def test_transformer_sharded():
    xr.set_device_type("TT")
    torch.manual_seed(42)

    loader = ModelLoader(ModelVariant.LTX2_TRANSFORMER)
    model = loader.load_model(dtype_override=torch.bfloat16)
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    mesh_shape, mesh_names = loader.get_mesh_config(xr.global_runtime_device_count())
    mesh = get_mesh(mesh_shape, mesh_names)

    run_graph_test(
        model,
        inputs,
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=loader.load_shard_spec,
    )
