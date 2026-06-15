# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""FLUX.1-dev — T5 text encoder component test (sequence embedding)."""

import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, run_graph_test

from third_party.tt_forge_models.flux.pytorch import ModelLoader, ModelVariant


@pytest.mark.xfail(
    reason="tt-mlir compiler INTERNAL error (RuntimeError: Bad StatusOr access: "
    "INTERNAL: Error code: 13) during graph lowering. IR shows a const-eval "
    "subgraph materializing a rank-0 scalar (ttnn.full shape=<>, tile layout); "
    "the scalar const-eval path errors out. Compiler-level issue, not a loader "
    "bug. Tracking issue TBD."
)
@pytest.mark.single_device
@pytest.mark.nightly
@pytest.mark.model_test
def test_t5_text_encoder():
    xr.set_device_type("TT")
    torch.manual_seed(42)

    loader = ModelLoader(ModelVariant.T5_TEXT_ENCODER)
    model = loader.load_model(dtype_override=torch.bfloat16)
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    run_graph_test(
        model,
        inputs,
        framework=Framework.TORCH,
    )
