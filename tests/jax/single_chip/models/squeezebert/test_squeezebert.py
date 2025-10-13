# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

import jax
import pytest
from flax import linen as nn
from infra import ComparisonConfig, JaxModelTester, RunMode
from jaxtyping import PyTree
from utils import BringupStatus, Category, ExecutionPass, failed_ttmlir_compilation

from third_party.tt_forge_models.config import Parallelism
from third_party.tt_forge_models.squeezebert.masked_lm.jax import (
    ModelLoader,
    ModelVariant,
)

VARIANT_NAME = ModelVariant.BASE
MODEL_INFO = ModelLoader._get_model_info(VARIANT_NAME)

# ----- Tester -----


class SqueezeBertTester(JaxModelTester):
    """Tester for SqueezeBERT model on a masked language modeling task"""

    def __init__(
        self,
        variant_name: ModelVariant,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ):
        self._model_loader = ModelLoader(variant_name)
        super().__init__(comparison_config, run_mode)

    # @override
    def _get_model(self) -> nn.Module:
        return self._model_loader.load_model()

    # @override
    def _get_forward_method_name(self):
        return "apply"

    # @override
    def _get_input_activations(self) -> Dict[str, jax.Array]:
        return self._model_loader.load_inputs()

    # @override
    def _get_input_parameters(self) -> PyTree:
        return self._model_loader.load_parameters()

    def _get_forward_method_args(self) -> list:
        return []

    # @override
    def _get_forward_method_kwargs(self) -> Dict[str, PyTree]:
        return {
            "variables": self._input_parameters,
            **self._input_activations,
            "train": False if self._run_mode == RunMode.INFERENCE else True,
            "rngs": (
                {"dropout": jax.random.key(1)}
                if self._run_mode == RunMode.TRAINING
                else None
            ),
        }


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> SqueezeBertTester:
    return SqueezeBertTester(VARIANT_NAME)


@pytest.fixture
def training_tester() -> SqueezeBertTester:
    return SqueezeBertTester(VARIANT_NAME, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    parallelism=Parallelism.SINGLE_DEVICE,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_squeezebert_inference(inference_tester: SqueezeBertTester):
    inference_tester.test()


@pytest.mark.training
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    parallelism=Parallelism.SINGLE_DEVICE,
    run_mode=RunMode.TRAINING,
    execution_pass=ExecutionPass.BACKWARD,
    bringup_status=BringupStatus.FAILED_TTMLIR_COMPILATION,
)
@pytest.mark.xfail(
    reason=failed_ttmlir_compilation(
        "error: failed to legalize operation 'ttir.convolution' "
        "https://github.com/tenstorrent/tt-mlir/issues/5307"
    )
)
def test_squeezebert_training(training_tester: SqueezeBertTester):
    training_tester.test()
