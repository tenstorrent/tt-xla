# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from infra import RunMode
from infra.evaluators import ComparisonConfig, PccConfig
from utils import BringupStatus, Category

from third_party.tt_forge_models.wan.pytorch import ModelLoader, ModelVariant

from .tester import WanVAETester

# ----- Common params -----

_WAN21_PARAMS = [
    pytest.param(
        ModelVariant.WAN21_T2V_14B,
        marks=[
            pytest.mark.record_test_properties(
                category=Category.MODEL_TEST,
                model_info=ModelLoader.get_model_info(ModelVariant.WAN21_T2V_14B),
                run_mode=RunMode.INFERENCE,
                bringup_status=BringupStatus.PASSED,
            ),
        ],
        id="wan21_t2v_14b",
    ),
]


def _make_tester(variant, vae_part: str) -> WanVAETester:
    return WanVAETester(
        variant,
        vae_part=vae_part,
        comparison_config=ComparisonConfig(pcc=PccConfig()),
    )


# ----- Fixtures -----


@pytest.fixture(params=_WAN21_PARAMS)
def encoder_tester(request) -> WanVAETester:
    return _make_tester(request.param, "encoder")


@pytest.fixture(params=_WAN21_PARAMS)
def decoder_tester(request) -> WanVAETester:
    return _make_tester(request.param, "decoder")


# ----- Tests -----


@pytest.mark.nightly
@pytest.mark.single_device
def test_torch_wan_vae_encoder_inference(encoder_tester: WanVAETester):
    encoder_tester.test()


@pytest.mark.nightly
@pytest.mark.single_device
def test_torch_wan_vae_decoder_inference(decoder_tester: WanVAETester):
    decoder_tester.test()
