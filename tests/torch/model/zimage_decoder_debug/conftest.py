# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for Z-Image VAE decoder debug tests."""

import pytest

from .shared import DTYPE, build_stage_specs, load_vae_and_latents


@pytest.fixture(scope="module")
def vae_decoder_context():
    """Load VAE once per module and precompute CPU stage activations."""
    vae, latents, stages = load_vae_and_latents(DTYPE)
    specs = build_stage_specs(vae)
    return {
        "vae": vae,
        "latents": latents,
        "stages": stages,
        "specs": specs,
        "dtype": DTYPE,
    }


def pytest_generate_tests(metafunc):
    if "stage_name" in metafunc.fixturenames:
        # Parametrize at collection time using a lightweight vae load is too slow;
        # use static stage names matching build_stage_specs order.
        names = [
            "latent_preprocess",
            "conv_in",
            "mid_block",
            "up_block_0",
            "up_block_1",
            "up_block_2",
            "up_block_3",
            "decoder_head",
            "prefix_through_up_0",
            "prefix_through_up_1",
            "prefix_through_up_2",
            "prefix_through_up_3",
            "prefix_through_up_4",
            "prefix_full_decoder",
        ]
        metafunc.parametrize("stage_name", names)
