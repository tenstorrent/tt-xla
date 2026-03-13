# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from infra import RunMode
from infra.evaluators import ComparisonConfig, PccConfig
from utils import BringupStatus, Category

from third_party.tt_forge_models.wan.pytorch import ModelLoader, ModelVariant

from .tester import WanTextEncoderTester

# ----- Common params -----

_WAN21_PARAMS = [
    pytest.param(
        ModelVariant.WAN21_T2V_13B,
        marks=[
            pytest.mark.record_test_properties(
                category=Category.MODEL_TEST,
                model_info=ModelLoader.get_model_info(ModelVariant.WAN21_T2V_13B),
                run_mode=RunMode.INFERENCE,
                bringup_status=BringupStatus.PASSED,
            ),
        ],
        id="wan21_t2v_13b",
    ),
]


def _make_tester(variant) -> WanTextEncoderTester:
    return WanTextEncoderTester(
        variant,
        comparison_config=ComparisonConfig(pcc=PccConfig()),
    )


# ----- Fixtures -----


@pytest.fixture(params=_WAN21_PARAMS)
def text_encoder_tester(request) -> WanTextEncoderTester:
    return _make_tester(request.param)


# ----- Tests -----


@pytest.mark.nightly
@pytest.mark.single_device
def test_torch_wan_text_encoder_inference(text_encoder_tester: WanTextEncoderTester):
    text_encoder_tester.test()
