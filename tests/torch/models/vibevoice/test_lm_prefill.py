# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""VibeVoice-1.5B — Qwen2.5 decoder prefill + LM head component test.

Params: ~1.54 B. This is the backbone the landed logits-only bringup already
covered; the test here differs by running it with **pretrained** weights at the
128-token conditioning-prompt length the TTS path actually produces.
"""

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test

from third_party.tt_forge_models.vibevoice.pytorch import ModelLoader, ModelVariant


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.single_device
def test_lm_prefill():
    xr.set_device_type("TT")
    torch.manual_seed(42)

    loader = ModelLoader(ModelVariant.LM_PREFILL)
    model = loader.load_model(dtype_override=torch.bfloat16)
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    run_graph_test(model, inputs, framework=Framework.TORCH)
