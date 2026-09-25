# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""VibeVoice-1.5B — acoustic tokenizer encoder component test. Params: ~344 M.

Encodes the full 222480-sample voice prompt into 70 acoustic latent frames
(3200:1 compression), which is the length the real conditioning path produces.

The encode is **chunked**, in 32000-sample pieces carrying upstream's streaming
conv cache across the boundaries. One pass over the whole prompt overflows L1 in
``ttnn.pad`` — 8115264 B of circular buffers against a 1499136 B limit — and the
boundary is an input-length one: 3200, 12800 and 32000 samples all pass, 64000
does not. The chunking lives in the loader's wrapper rather than in this test so
the e2e path gets it too, and it is exact rather than approximate: against a
single full-length encode it reads PCC 1.00000000 on CPU, where chunking the
naive way (each chunk independent) reads 0.9616.
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
def test_acoustic_encoder():
    xr.set_device_type("TT")
    torch.manual_seed(42)

    loader = ModelLoader(ModelVariant.ACOUSTIC_ENCODER)
    model = loader.load_model(dtype_override=torch.bfloat16)
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    run_graph_test(model, inputs, framework=Framework.TORCH)
