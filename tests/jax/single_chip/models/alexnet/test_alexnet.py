# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Sequence

import jax
import pytest
from flax import linen as nn
from infra import ComparisonConfig, Framework, ModelTester, RunMode

from tests.utils import (
    BringupStatus,
    Category,
    ModelGroup,
    ModelSource,
    ModelTask,
    build_model_name,
    failed_ttmlir_compilation,
)

from .model_implementation import AlexNetModel

MODEL_NAME = build_model_name(
    Framework.JAX,
    "alexnet",
    None,
    ModelTask.CV_IMAGE_CLS,
    ModelSource.CUSTOM,
)


# ----- Tester -----


class AlexNetTester(ModelTester):
    """Tester for AlexNet CNN model."""

    def __init__(
        self,
        model_class,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        self._model_class = model_class
        super().__init__(comparison_config, run_mode)

    # @override
    def _get_model(self) -> nn.Module:
        return self._model_class()

    # @override
    def _get_forward_method_name(self) -> str:
        return "apply"

    # @override
    def _get_input_activations(self) -> Sequence[jax.Array]:
        prng_key = jax.random.PRNGKey(23)
        img = jax.random.randint(
            key=prng_key,
            # B, H, W, C
            shape=(4, 224, 224, 3),
            # In the original paper inputs are normalized with individual channel
            # values learned from training set.
            minval=-128,
            maxval=128,
        )
        return img

    # @override
    def _get_forward_method_args(self):
        inp = self._get_input_activations()

        # Example of flax.linen convention of first instatiating a model object
        # and then later calling init to generate a set of initial tensors (parameters
        # and maybe some extra state). Parameters are not stored with the models
        # themselves, they are provided together with inputs to the forward method.
        parameters = self._model.init(jax.random.PRNGKey(42), inp, train=False)

        return [parameters, inp]

    # @override
    def _get_forward_method_kwargs(self) -> Dict[str, jax.Array]:
        return {"train": False}

    # @override
    def _get_static_argnames(self):
        return ["train"]


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> AlexNetTester:
    return AlexNetTester(AlexNetModel)


@pytest.fixture
def training_tester() -> AlexNetTester:
    return AlexNetTester(AlexNetModel, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.FAILED_TTMLIR_COMPILATION,
)
@pytest.mark.xfail(
    reason=failed_ttmlir_compilation(
        "failed to legalize operation 'ttir.gather' "
        "https://github.com/tenstorrent/tt-xla/issues/318"
    )
)
def test_alexnet_inference(inference_tester: AlexNetTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_alexnet_training(training_tester: AlexNetTester):
    training_tester.test()
