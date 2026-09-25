# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""LTX-2 — LTX2TextConnectors (connectors) component test (1.43B)."""

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test

from third_party.tt_forge_models.ltx2.pytorch import ModelLoader, ModelVariant


@pytest.mark.xfail(
    reason="tt/torch_xla cannot lower the LTX-2 rope head-merge: "
    "apply_split_rotary_emb (transformer_ltx2.py) does "
    "out.swapaxes(1, 2).reshape(b, t, -1) and _assert_tensor_metadata rejects "
    "reshaping the transposed (non-contiguous) tensor: 'Cannot view a tensor "
    "with shape [1,128,30,128] strides (491520,128,16384,1) as (1,128,3840)'. "
    ".contiguous() is a no-op on XLA lazy tensors, so it is not fixable in the "
    "loader. Shared by connectors + transformer — "
    "https://github.com/tenstorrent/tt-xla/issues/5196"
)
@pytest.mark.model_test
@pytest.mark.single_device
def test_connectors():
    xr.set_device_type("TT")
    torch.manual_seed(42)

    loader = ModelLoader(ModelVariant.LTX2_CONNECTORS)
    model = loader.load_model(dtype_override=torch.float32)
    inputs = loader.load_inputs(dtype_override=torch.float32)

    run_graph_test(model, inputs, framework=Framework.TORCH)
