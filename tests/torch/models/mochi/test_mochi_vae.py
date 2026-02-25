# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from infra import RunMode
from utils import BringupStatus, Category

from third_party.tt_forge_models.mochi.pytorch import ModelLoader, ModelVariant

from .tester import MochiVAETester

# ----- Constants -----

_DRAM_OOM_SKIP_REASON = (
    "DRAM OOM: fails due to https://github.com/tenstorrent/tt-xla/issues/2973"
)

# ----- Fixtures -----


@pytest.fixture(
    params=[
        pytest.param(
            ModelVariant.MOCHI,
            marks=[
                pytest.mark.record_test_properties(
                    category=Category.MODEL_TEST,
                    model_info=ModelLoader.get_model_info(ModelVariant.MOCHI),
                    run_mode=RunMode.INFERENCE,
                    bringup_status=BringupStatus.FAILED_RUNTIME,
                ),
            ],
            id="mochi",
        ),
        pytest.param(
            ModelVariant.MOCHI_TILED,
            marks=[
                pytest.mark.skip(reason=_DRAM_OOM_SKIP_REASON),
                pytest.mark.record_test_properties(
                    category=Category.MODEL_TEST,
                    model_info=ModelLoader.get_model_info(ModelVariant.MOCHI_TILED),
                    run_mode=RunMode.INFERENCE,
                    bringup_status=BringupStatus.FAILED_RUNTIME,
                ),
            ],
            id="mochi_tiled",
        ),
    ]
)
def inference_tester(request) -> MochiVAETester:
    return MochiVAETester(request.param)


# ----- Tests -----


@pytest.mark.nightly
@pytest.mark.single_device
def test_torch_mochi_vae_decoder_inference(inference_tester: MochiVAETester):
    inference_tester.test()
