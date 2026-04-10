# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode
from utils import BringupStatus, Category, incorrect_result

from tests.infra.testers.compiler_config import CompilerConfig
from third_party.tt_forge_models.gemma4.causal_lm.pytorch import (
    ModelLoader,
    ModelVariant,
)

from .tester import Gemma4Tester

"""Run Gemma4 tests across all available variants."""


# ----- Variant Classification -----

# Small variants that fit on a single device
_SINGLE_DEVICE_VARIANTS = [
    ModelVariant.GEMMA_4_E2B,
    ModelVariant.GEMMA_4_E2B_IT,
    ModelVariant.GEMMA_4_E4B,
    ModelVariant.GEMMA_4_E4B_IT,
]

# 26B MoE variants
_MOE_VARIANTS = [
    ModelVariant.GEMMA_4_26B_A4B,
    ModelVariant.GEMMA_4_26B_A4B_IT,
]

# 31B dense variants - sharded across multiple devices
_31B_VARIANTS = [
    ModelVariant.GEMMA_4_31B,
    ModelVariant.GEMMA_4_31B_IT,
]


# ----- Fixtures -----


def _single_device_variant_param(v):
    """Create a pytest parameter for a single-device variant."""
    model_info = ModelLoader.get_model_info(v)
    bringup_status = BringupStatus.INCORRECT_RESULT

    marks = [
        pytest.mark.model_test,
        pytest.mark.xfail(
            reason=incorrect_result("Low PCC for Gemma4 causal LM inference")
        ),
        pytest.mark.record_test_properties(
            category=Category.MODEL_TEST,
            model_info=model_info,
            run_mode=RunMode.INFERENCE,
            bringup_status=bringup_status,
        ),
    ]

    # E4B variants are larger, mark them
    if v in (ModelVariant.GEMMA_4_E4B, ModelVariant.GEMMA_4_E4B_IT):
        marks.append(pytest.mark.large)

    return pytest.param((v, bringup_status), marks=tuple(marks))


def _moe_variant_param(v):
    """Create a pytest parameter for a 26B MoE variant."""
    model_info = ModelLoader.get_model_info(v)
    bringup_status = BringupStatus.INCORRECT_RESULT

    marks = [
        pytest.mark.model_test,
        pytest.mark.large,
        pytest.mark.xfail(
            reason=incorrect_result("Low PCC for Gemma4 26B MoE causal LM inference")
        ),
        pytest.mark.record_test_properties(
            category=Category.MODEL_TEST,
            model_info=model_info,
            run_mode=RunMode.INFERENCE,
            bringup_status=bringup_status,
        ),
    ]

    return pytest.param((v, bringup_status), marks=tuple(marks))


def _31b_variant_param(v):
    """Create a pytest parameter for a 31B dense variant."""
    model_info = ModelLoader.get_model_info(v)
    bringup_status = BringupStatus.INCORRECT_RESULT

    marks = [
        pytest.mark.model_test,
        pytest.mark.large,
        pytest.mark.xfail(
            reason=incorrect_result("Low PCC for Gemma4 31B causal LM inference")
        ),
        pytest.mark.record_test_properties(
            category=Category.MODEL_TEST,
            model_info=model_info,
            run_mode=RunMode.INFERENCE,
            bringup_status=bringup_status,
        ),
    ]

    return pytest.param((v, bringup_status), marks=tuple(marks))


_SINGLE_DEVICE_PARAMS = [
    _single_device_variant_param(v) for v in _SINGLE_DEVICE_VARIANTS
]
_SINGLE_DEVICE_IDS = [v.name.lower() for v in _SINGLE_DEVICE_VARIANTS]

_MOE_PARAMS = [_moe_variant_param(v) for v in _MOE_VARIANTS]
_MOE_IDS = [v.name.lower() for v in _MOE_VARIANTS]

_31B_PARAMS = [_31b_variant_param(v) for v in _31B_VARIANTS]
_31B_IDS = [v.name.lower() for v in _31B_VARIANTS]


def _create_tester(variant) -> Gemma4Tester:
    compiler_config = CompilerConfig(experimental_weight_dtype="bfp_bf8")
    return Gemma4Tester(variant, compiler_config=compiler_config)


def _create_tp_tester(variant) -> Gemma4Tester:
    compiler_config = CompilerConfig(experimental_weight_dtype="bfp_bf8")
    return Gemma4Tester(variant, compiler_config=compiler_config)


@pytest.fixture(params=_SINGLE_DEVICE_PARAMS, ids=_SINGLE_DEVICE_IDS)
def single_device_inference_tester(request) -> Gemma4Tester:
    """Fixture for single-device Gemma4 inference testing."""
    variant, bringup_status = request.param
    request.node.bringup_status = bringup_status
    return _create_tester(variant)


@pytest.fixture(params=_MOE_PARAMS, ids=_MOE_IDS)
def moe_inference_tester(request) -> Gemma4Tester:
    """Fixture for 26B MoE Gemma4 inference testing."""
    variant, bringup_status = request.param
    request.node.bringup_status = bringup_status
    return _create_tp_tester(variant)


@pytest.fixture(params=_31B_PARAMS, ids=_31B_IDS)
def dense_31b_inference_tester(request) -> Gemma4Tester:
    """Fixture for 31B dense Gemma4 inference testing."""
    variant, bringup_status = request.param
    request.node.bringup_status = bringup_status
    return _create_tp_tester(variant)


# ----- Tests -----


@pytest.mark.nightly
@pytest.mark.single_device
def test_torch_gemma4_inference(single_device_inference_tester: Gemma4Tester):
    single_device_inference_tester.test()


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.forked
def test_torch_gemma4_inference_26b_moe(moe_inference_tester: Gemma4Tester):
    moe_inference_tester.test()


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.forked
def test_torch_gemma4_inference_31b(dense_31b_inference_tester: Gemma4Tester):
    dense_31b_inference_tester.test()
