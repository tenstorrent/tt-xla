# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from infra import RunMode
from utils import BringupStatus, Category

from third_party.tt_forge_models.mochi.pytorch import ModelLoader, ModelVariant

from .tester import MochiVAETester

# ----- Fixtures -----


def _variant_param(v):
    """Create a pytest parameter for each ModelVariant with bringup_status and marks."""
    marks = []

    model_info = ModelLoader.get_model_info(v)

    marks.extend(
        [
            pytest.mark.model_test,
            pytest.mark.large,
            pytest.mark.record_test_properties(
                category=Category.MODEL_TEST,
                model_info=model_info,
                run_mode=RunMode.INFERENCE,
                bringup_status=BringupStatus.PASSED,
            ),
        ]
    )

    return pytest.param((v, BringupStatus.PASSED), marks=tuple(marks))


_MOCHI_VAE_PARAMS = [_variant_param(v) for v in list(ModelVariant)]
_MOCHI_VAE_IDS = [v.name.lower() for v in list(ModelVariant)]


@pytest.fixture(params=_MOCHI_VAE_PARAMS, ids=_MOCHI_VAE_IDS)
def inference_tester(request) -> MochiVAETester:
    """Fixture that returns a MochiVAETester configured for each model variant."""
    variant, bringup_status = request.param
    request.node.bringup_status = bringup_status
    return MochiVAETester(variant)


# ----- Tests -----


@pytest.mark.nightly
@pytest.mark.single_device
def test_torch_mochi_vae_decoder_inference(inference_tester: MochiVAETester):
    """
    Test Mochi VAE decoder inference on TT hardware.

    This test validates that:
    - Model loads from HuggingFace
    - Compilation succeeds with TT backend
    - Execution completes without errors
    - Output shape is correct

    Runs for both tiled and non-tiled variants.
    """
    inference_tester.test()
