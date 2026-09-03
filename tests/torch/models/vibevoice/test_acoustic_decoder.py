# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""VibeVoice-1.5B — acoustic tokenizer decoder component test. Params: ~344 M.

Decodes one acoustic latent frame into a 3200-sample waveform chunk; the
generation loop calls this once per emitted frame.

Runs in **float32**, not bfloat16, deliberately. In bfloat16 this component
lands at PCC 0.988, just under the 0.99 gate; in float32 it is 0.998. The
error is precision in a deep causal-conv stack rather than a compiler defect,
so the fix is the dtype rather than a weakened threshold.
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
def test_acoustic_decoder():
    xr.set_device_type("TT")
    torch.manual_seed(42)

    loader = ModelLoader(ModelVariant.ACOUSTIC_DECODER)
    model = loader.load_model(dtype_override=torch.float32)
    inputs = loader.load_inputs(dtype_override=torch.float32)

    run_graph_test(model, inputs, framework=Framework.TORCH)
