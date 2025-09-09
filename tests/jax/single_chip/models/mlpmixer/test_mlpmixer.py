# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


import flax.traverse_util
import fsspec
import jax
import ml_collections
import numpy
import pytest
from flax import linen as nn
from infra import Framework, JaxModelTester, RunMode
from jaxtyping import PyTree
from utils import (
    BringupStatus,
    Category,
    ModelGroup,
    ModelSource,
    ModelTask,
    build_model_name,
    incorrect_result,
)

from .model_implementation import MlpMixer

MODEL_NAME = build_model_name(
    Framework.JAX,
    "mlpmixer",
    None,
    ModelTask.CV_IMAGE_CLS,
    ModelSource.CUSTOM,
)

# Hyperparameters for Mixer-B/16
patch_size = 16
num_classes = 21843
num_blocks = 12
hidden_dim = 768
token_mlp_dim = 384
channel_mlp_dim = 3072


class MlpMixerTester(JaxModelTester):
    """Tester for MlpMixer model."""

    # @override
    def _get_model(self) -> nn.Module:
        patch = ml_collections.ConfigDict({"size": (patch_size, patch_size)})
        model = MlpMixer(
            patches=patch,
            num_classes=num_classes,
            num_blocks=num_blocks,
            hidden_dim=hidden_dim,
            tokens_mlp_dim=token_mlp_dim,
            channels_mlp_dim=channel_mlp_dim,
        )
        link = "https://storage.googleapis.com/mixer_models/imagenet21k/Mixer-B_16.npz"
        with fsspec.open("filecache::" + link, cache_storage="/tmp/files/") as f:
            weights = numpy.load(f, encoding="bytes")
            state_dict = {k: v for k, v in weights.items()}
            model.params = {"params": flax.traverse_util.unflatten_dict(state_dict, sep="/")}

        return model

    # @override
    def _get_input_activations(self) -> jax.Array:
        key = jax.random.PRNGKey(42)
        random_image = jax.random.normal(key, (1, 224, 224, 3))
        return random_image


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> MlpMixerTester:
    return MlpMixerTester()


@pytest.fixture
def training_tester() -> MlpMixerTester:
    return MlpMixerTester(RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.push
@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.INCORRECT_RESULT,
)
@pytest.mark.xfail(
    reason=incorrect_result(
        "Atol comparison failed. Calculated: atol=15.194854736328125. Required: atol=0.16 "
        "https://github.com/tenstorrent/tt-xla/issues/379"
    )
)
def test_mlpmixer_inference(inference_tester: MlpMixerTester):
    inference_tester.test()


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_mlpmixer_training(training_tester: MlpMixerTester):
    training_tester.test()
