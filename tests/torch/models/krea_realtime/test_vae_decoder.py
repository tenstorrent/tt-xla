# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Krea Realtime — AutoencoderKLWan (3D causal VAE) decoder component test.

IN:  z       (1, 16, 3, 60, 104)   bfloat16
OUT: sample  (1, 3, 9, 480, 832)   bfloat16
"""

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test

from tests.torch.models.krea_realtime.shared import (
    DTYPE,
    LATENT_H,
    LATENT_W,
    NUM_CHANNELS_LATENTS,
    NUM_LATENT_FRAMES,
    load_vae,
)


class VAEDecoderWrapper(torch.nn.Module):
    """Wrap AutoencoderKLWan so the forward signature is just (z) -> tensor.

    Default `vae(z)` runs encode+decode and returns a model-output object;
    we want only the decoder, with a plain tensor out.
    """

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        return self.vae.decode(z, return_dict=False)[0]


@pytest.mark.xfail(
    reason="VAE temporal slice fails on TT (out-of-range slice on size-1 dim) — https://github.com/tenstorrent/tt-xla/issues/4465"
)
def test_vae_decoder():
    xr.set_device_type("TT")
    torch.manual_seed(42)

    wrapper = VAEDecoderWrapper(load_vae()).eval()

    z = torch.randn(
        1,
        NUM_CHANNELS_LATENTS,
        NUM_LATENT_FRAMES,
        LATENT_H,
        LATENT_W,
        dtype=DTYPE,
    )

    run_graph_test(
        wrapper,
        [z],
        framework=Framework.TORCH,
    )
