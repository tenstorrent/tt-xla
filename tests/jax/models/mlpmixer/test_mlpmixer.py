# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Dict, Sequence

import flax.traverse_util
import fsspec
import jax
import ml_collections
import numpy
import pytest
from flax import linen as nn
from infra import ModelTester, RunMode
from utils import record_model_test_properties, runtime_fail

from .model_implementation import MlpMixer

# Hyperparameters for Mixer-B/16
patch_size = 16
num_classes = 21843
num_blocks = 12
hidden_dim = 768
token_mlp_dim = 384
channel_mlp_dim = 3072


class MlpMixerTester(ModelTester):
    """Tester for MlpMixer model."""

    # @override
    def _get_model(self) -> nn.Module:
        patch = ml_collections.ConfigDict({"size": (patch_size, patch_size)})
        return MlpMixer(
            patches=patch,
            num_classes=num_classes,
            num_blocks=num_blocks,
            hidden_dim=hidden_dim,
            tokens_mlp_dim=token_mlp_dim,
            channels_mlp_dim=channel_mlp_dim,
        )

    @staticmethod
    def _retrieve_pretrained_weights() -> Dict:
        # TODO(stefan): Discuss how weights should be handled org wide
        link = "https://storage.googleapis.com/mixer_models/imagenet21k/Mixer-B_16.npz"
        with fsspec.open("filecache::" + link, cache_storage="/tmp/files/") as f:
            weights = numpy.load(f, encoding="bytes")
            state_dict = {k: v for k, v in weights.items()}
            pytree = flax.traverse_util.unflatten_dict(state_dict, sep="/")
        return {"params": pytree}

    # @override
    def _get_forward_method_name(self) -> str:
        return "apply"

    # @override
    def _get_input_activations(self) -> jax.Array:
        key = jax.random.PRNGKey(42)
        random_image = jax.random.normal(key, (1, 224, 224, 3))
        return random_image

    # @override
    def _get_forward_method_args(self) -> Sequence[Any]:
        ins = self._get_input_activations()
        weights = self._retrieve_pretrained_weights()

        # Alternatively, weights could be randomly initialized like this:
        # weights = self._model.init(jax.random.PRNGKey(42), ins)

        # JAX frameworks have a convention of passing weights as the first argument
        return [weights, ins]


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> MlpMixerTester:
    return MlpMixerTester()


@pytest.fixture
def training_tester() -> MlpMixerTester:
    return MlpMixerTester(RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.skip(
    reason=runtime_fail(
        "Statically allocated circular buffers in program 16 clash with L1 buffers "
        "on core range [(x=0,y=0) - (x=6,y=0)]. L1 buffer allocated at 475136 and "
        "static circular buffer region ends at 951136 "
        "(https://github.com/tenstorrent/tt-xla/issues/187)"
    )
)  # segfault
def test_mlpmixer_inference(
    inference_tester: MlpMixerTester,
    record_tt_xla_property: Callable,
):
    record_model_test_properties(record_tt_xla_property, "mlpmixer")

    inference_tester.test()


@pytest.mark.skip(reason="Support for training not implemented")
def test_mlpmixer_training(
    training_tester: MlpMixerTester,
    record_tt_xla_property: Callable,
):
    record_model_test_properties(record_tt_xla_property, "mlpmixer")

    training_tester.test()
