# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Dict, Sequence


import jax
import jax.numpy as jnp
import numpy
import pytest
import fsspec
from flax import linen as nn
from infra import ModelTester, RunMode
from .model_implementation import MlpMixer
from .util import build_pytee_from_npy


# hypers
patch_size = 16
num_classes = 21843
num_blocks = 12
hidden_dim = 768
token_mlp_dim = 384
channel_mlp_dim = 3072


def Mixer_B_16_pretrained():
    # TODO(stefan): Discuss how weights should be handled org wide
    link = "https://storage.googleapis.com/mixer_models/imagenet21k/Mixer-B_16.npz"
    with fsspec.open("filecache::" + link, cache_storage="/tmp/files/") as f:
        weights = numpy.load(f, encoding="bytes")
        pytree = build_pytee_from_npy(weights)
    return pytree


class MlpMixerTester(ModelTester):
    """Tester for MlpMixer model."""

    # @override
    def _get_model(self) -> nn.Module:
        patch = jnp.ones((patch_size, patch_size))
        return MlpMixer(
            patches=patch,
            num_classes=num_classes,
            num_blocks=num_blocks,
            hidden_dim=hidden_dim,
            tokens_mlp_dim=token_mlp_dim,
            channels_mlp_dim=channel_mlp_dim,
        )

    # @override
    def _get_forward_method_name(self) -> str:
        return "apply"

    # @override
    def _get_input_activations(self) -> Sequence[jax.Array]:
        key = jax.random.PRNGKey(42)
        random_image = jax.random.normal(key, (1, 196, 196, 3))
        return random_image

    # @override
    def _get_forward_method_args(self):
        ins = self._get_input_activations()
        weights = Mixer_B_16_pretrained()
        # Required to bypass "Initializer expected to generate shape (16, 16, 3, 768) but got shape (256, 3, 768)"
        kernel = weights["params"]["stem"]["kernel"]
        kernel = kernel.reshape(-1, 3, hidden_dim)
        weights["params"]["stem"]["kernel"] = kernel

        # Alternatively, weights could be randomly initialized like this:
        # weights = self._model.init(jax.random.PRNGKey(42), ins, train=False)

        return [weights, ins]

    # @override
    def _get_forward_method_kwargs(self) -> Dict[str, jax.Array]:
        return {"train": False}


# ----- Fixtures -----
@pytest.fixture
def inference_tester() -> MlpMixerTester:
    return MlpMixerTester()


@pytest.fixture
def training_tester() -> MlpMixerTester:
    return MlpMixerTester(RunMode.TRAINING)


# ----- Tests -----
@pytest.mark.skip(
    reason="error: failed to legalize operation 'ttir.convolution' that was explicitly marked illegal"
)
def test_mlpmixer(inference_tester: MlpMixerTester):
    inference_tester.test()


@pytest.mark.skip(reason="Support for training not implemented")
def test_mlpmixer_training(training_tester: MlpMixerTester):
    training_tester.test()
